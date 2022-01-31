
import re
import os
import h5py
import glob
import time
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import Atom, MoleculeGraph

class HDF5Dataset(Dataset):
    '''
    Pytorch dataset for AFM data using HDF5 database.

    Arguments:
        hdf5_path: str. Path to HDF5 database file.
        mode: 'train', 'val', or 'test'. Which dataset to use.
    '''
    def __init__(self, hdf5_path, mode='train'):
        self.hdf5_path = hdf5_path
        self.mode = mode
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(f'mode should be one of "train", "val", or "test", but got {self.mode}')
        
    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            length = len(f[self.mode]['X'])
        return length
        
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f[self.mode]
            X = dataset['X'][idx]
            Y = dataset['Y'][idx] if 'Y' in dataset.keys() else []
            xyz = self._unpad_xyz(dataset['xyz'][idx])
        return X, Y, xyz

    def _unpad_xyz(self, xyz):
        return xyz[xyz[:,-1] > 0]

def _worker_init_fn(worker_id):
    np.random.seed(int((time.time() % 1e5)*1000) + worker_id)

class _collate_wrapper():

    def __init__(self, collate_fn, preproc_fn):
        self.collate_fn = collate_fn
        self.preproc_fn = preproc_fn

    def __call__(self, batch):

        # Combine samples into a batch
        Xs = []
        Ys = []
        xyzs = []
        for X, Y, xyz in batch:
            Xs.append(X)
            Ys.append(Y)
            xyzs.append(xyz)
        Xs = list(np.stack(Xs, axis=0).transpose(1, 0, 2, 3, 4))
        if len(Ys[0]) > 0:
            Ys = list(np.stack(Ys, axis=0).transpose(1, 0, 2, 3))

        # Run preprocessing and collate
        batch = (Xs, Ys, xyzs)
        batch = self.preproc_fn(batch)
        batch = self.collate_fn(batch)

        return batch

def collate(batch):
    '''
    Collate graph samples into a batch.

    Arguments:
        batch: tuple (X, mols, removed, xyz, ref_graphs, ref_dist, ref_atoms, box_borders)
            X: np.ndarray of shape (batch_size, x, y, z). Input AFM image.
            mols: list of MoleculeGraph. Input molecules.
            removed: list of tuples (atom, bonds), where atom is an Atom object and bonds is a list of
                0s and 1s indicating the existence of bond connection to atoms in mols.
            xyz: list of np.ndarray of shape (num_atoms, 5). List of original molecules.
            ref_graphs: list of MoleculeGraph. Complete reference graphs before atom removals.
            ref_dist: np.ndarray of shape (batch_size, x, y, z). Reference position distribution.
            ref_atoms: list of Atom. Reference atom positions and classes for teacher forcing.
            box_borders: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent of the
                position distribution region in angstroms.
    
    Returns: tuple (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs, box_borders)
        X: torch.Tensor of shape (batch_size, 1, x, y, z). Input AFM images.
        node_inputs: torch.Tensor of shape (total_atoms, 3+n_classes), where total_atoms is the total number
            of atoms in input molecules. Graph node inputs.
        edges: torch.Tensor of shape (2, total_edges), where total_edges is the total number of edge connections
            in input molecules. Edge connections between input nodes.
        node_rem: list of torch.Tensor of shape (n_nodes, 4). Coordinates and classes of nodes to be predicted. Each batch
            item has a varying number of nodes n_nodes that can be predicted.
        edge_rem: list of torch.Tensor of shape (num_atoms,). Indicators for bond connections of the node to be predicted.
        terminate: torch.Tensor of shape (batch_size,). Indicator list for whether the molecule graph is complete.
        ref_dist: torch.Tensor of shape (batch_size, x, y, z). Reference position distribution.
        ref_atoms: torch.Tensor of shape (batch_size, 4). Reference atom positions and classes for teacher forcing.
        Ns: list in ints. Number of input nodes in each batch item.
        xyz: list of np.ndarray of shape (num_atoms, 5). Unchanged from input argument.
        ref_graphs: list of MoleculeGraph. Unchanged from input argument.
        box_borders: tuple. Unchanged from input argument.
    '''

    X, mols, removed, xyz, ref_graphs, ref_dist, ref_atoms, box_borders = batch
    assert len(X) == len(mols) == len(removed) == len(xyz) == len(ref_graphs) == len(ref_dist)

    mol_arrays = []
    edges = []
    edge_rem = []
    node_rem = []
    terminate = []
    ind_count = 0
    Ns = []

    for i, (mol, rem) in enumerate(zip(mols, removed)):

        if (mol_array := mol.array(xyz=True, class_weights=True)) != []:
            mol_arrays.append(mol_array)
        edges += [[b[0]+ind_count, b[1]+ind_count] for b in mol.bonds]

        if len(rem) > 0:
            e = []; n = []; t = 0
            for atom, bond_rem in rem:
                e.append(torch.tensor(bond_rem).float())
                n.append(atom.array(xyz=True, class_index=True))
        else:
            e = [torch.zeros(len(mol))]
            n = [np.zeros(4)]
            t = 1
        edge_rem.append(torch.from_numpy(np.stack(e, axis=0)).float())
        node_rem.append(torch.from_numpy(np.stack(n, axis=0)).float())
        terminate.append(t)

        ind_count += len(mol)
        Ns.append(len(mol))

    terminate = torch.tensor(terminate).float()

    X = torch.from_numpy(X).float()
    if X.ndim == 4:
        X = X.unsqueeze(1)

    ref_dist = torch.from_numpy(ref_dist).float()
        
    if len(mol_arrays) > 0:
        node_inputs = torch.from_numpy(np.concatenate(mol_arrays, axis=0)).float()
    else:
        node_inputs = torch.empty((0))
    edges = torch.tensor(edges).long().T

    ref_atoms = np.stack([r.array(xyz=True, class_index=True) if r else np.zeros(4)
        for r in ref_atoms], axis=0)
    ref_atoms = torch.from_numpy(ref_atoms).float()

    return X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs, box_borders

def uncollate(pred, batch):
    '''
    Convert graph batch back into separated format.
    Arguments:
        pred: tuple (pred_nodes, pred_edges, pred_terminate, pred_dist)
            pred_nodes: torch.Tensor of shape (batch_size, 3+n_classes). Predicted nodes.
            pred_edges: list of torch.Tensor of shape (num_atoms,). Predicted probabilities for edge connections.
            pred_dist: torch.Tensor of shape (batch_size, x, y, z). Predicted distribution for atom position.
        batch: tuple (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atom, Ns, xyz, ref_graphs).
            Same as return values of collate_grid.
    Returns: tuple (X, mols, pred, ref, xyz, ref_dist, ref_graph)
        X: np.ndarray of shape (batch_size, x, y, z). Input AFM image.
        mols: list of MoleculeGraph. Input molecules.
        pred: list of tuples (atom, bond, pred_dist), where atom is an Atom object, bond is a list,
            and pred_dist is np.ndarray of shape (x, y, z).
        ref: list of tuples (atom, bond, terminate, ref_dist), where atom is an Atom object, bond is a list,
            terminate is an int, and ref_dist is np.ndarray of shape (x, y, z).
        xyz: list of np.ndarray of shape (num_atoms, 5). Same as input xyz.
        ref_dist: np.ndarray of shape (batch_size, x, y, z). Reference position distribution.
        ref_graphs: list of MoleculeGraph. Unchanged from input.
    '''
    X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs = batch
    
    n_classes = pred[0].size(1) - 3

    X = X.squeeze().numpy()
    node_inputs = [n.numpy() for n in torch.split(node_inputs, split_size_or_sections=Ns)]
    edges = edges.numpy()

    node_rem = [n.numpy() for n in node_rem]
    edge_rem = [e.numpy() for e in edge_rem]
    terminate = terminate.numpy()
    ref_dist = ref_dist.numpy()
    ref_atoms = ref_atoms.numpy()

    pred_nodes, pred_edges, pred_dist = pred
    pred_class_weights = torch.nn.functional.softmax(pred_nodes[:,3:], dim=1).numpy()
    pred_xyz = pred_nodes[:,:3].numpy()
    pred_edges = [list(e.numpy()) for e in pred_edges]
    pred_dist = pred_dist.numpy()

    mols = []
    ref = []
    pred = []
    count = 0
    prev_ind = 0

    for i, N in enumerate(Ns):

        atoms = [Atom(a[:3], class_weights=a[3:]) for a in node_inputs[i]]
        bonds = []
        if edges.size > 0:
            ind = np.searchsorted(edges[0], count+N)
            for j in range(prev_ind, ind):
                bonds.append((edges[0,j]-count, edges[1,j]-count))
            prev_ind = ind
            count += N
        mols.append(MoleculeGraph(atoms, bonds))
        
        pred_atom = Atom(pred_xyz[i], class_weights=pred_class_weights[i])
        pred.append((pred_atom, pred_edges[i], pred_dist[i]))

        ref_pairs = []
        for a, b in zip(node_rem[i], edge_rem[i]):
            ref_atom = Atom(a[:3], class_weights=np.eye(n_classes)[int(a[3])])
            ref_bonds = [int(v) for v in b]
            ref_pairs.append((ref_atom, ref_bonds))
        ref.append((ref_pairs, int(terminate[i]), ref_dist[i]))

    return X, mols, pred, ref, xyz, ref_dist, pred_dist, ref_graphs, ref_atoms
    
def make_hdf5_dataloader(datadir, preproc_fn, collate_fn=collate, mode='train', batch_size=30,
    shuffle=True, num_workers=6, world_size=1, rank=0):
    '''
    Produce a dataset and dataloader from data directory.

    Arguments:
        hdf5_path: str. Path to HDF5 database file.
        preproc_fn: Python function. Preprocessing function to apply to each batch.
        collate_fn: Python function. Collate function that returns each batch.
        mode: 'train', 'val', or 'test'. Which dataset to use.
        batch_size: int. Number of samples in each batch.
        shuffle: bool. Whether to shuffle the sample order on each epoch.
        num_workers: int. Number of parallel processes for data loading.
        world_size: int. Number of parallel processes if using distributed training.
        rank: int. Index of current process if using distributed training.
    
    Returns: tuple (dataset, dataloader, sampler)
        dataset: HDF5Dataset.
        dataloader: DataLoader.
        sampler: DistributedSampler or None if world size == 1.
    '''
    dataset = HDF5Dataset(datadir, mode)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(mode == 'train'))
        shuffle = False
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_wrapper(collate_fn, preproc_fn),
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        timeout=300,
        pin_memory=True
    )
    return dataset, dataloader, sampler

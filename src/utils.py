
import os
import copy
import torch
import shutil
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

from PIL import Image
from scipy.stats import multivariate_normal
from skimage import feature, measure

from c.bindings import match_template_pool
from cuda.bindings import match_template as match_template_cuda, ccl, find_label_min

elements = ['H' , 'He', 
            'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne', 
            'Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar',
             'K', 'Ca', 
            'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr',
             'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                        'In', 'Sn', 'Sb', 'Te',  'I', 'Xe'
]

# Reference: http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bond_lengths = {
     1: { 1: 0.74,  6: 1.09,  7: 1.01,  8: 0.96,  9: 0.92, 14: 1.48, 15: 1.42, 16: 1.34, 17: 1.27, 35: 1.41},
     6: {           6: 1.54,  7: 1.47,  8: 1.43,  9: 1.33, 14: 1.86, 15: 1.87, 16: 1.81, 17: 1.77, 35: 1.94},
     7: {                     7: 1.46,  8: 1.44,  9: 1.39, 14: 1.72, 15: 1.77, 16: 1.68, 17: 1.91, 35: 2.14},
     8: {                               8: 1.48,  9: 1.42, 14: 1.61, 15: 1.60, 16: 1.51, 17: 1.64, 35: 1.72},
     9: {                                         9: 1.43, 14: 1.56, 15: 1.56, 16: 1.58, 17: 1.66, 35: 1.78},
    14: {                                                  14: 2.34, 15: 2.27, 16: 2.10, 17: 2.04, 35: 2.16},
    15: {                                                            15: 2.21, 16: 2.10, 17: 2.04, 35: 2.22},
    16: {                                                                      16: 2.04, 17: 2.01, 35: 2.25},
    17: {                                                                                17: 1.99, 35: 2.18},
    35: {                                                                                          35: 2.28}
}

vdW_radii = {
    1: 1.487,
    6: 1.908,
    7: 1.78,
    8: 1.661,
    9: 1.75,
    14: 1.9,
    15: 2.1,
    16: 2.0,
    17: 1.948,
    35: 2.22
}

def _calc_plot_dim(n, f=0.3):
    rows = max(int(np.sqrt(n) - f), 1)
    cols = 1
    while rows*cols < n:
        cols += 1
    return rows, cols

class Atom:
    '''
    A class representing an atom with a position, element and a charge.
    Arguments:
        xyz: list of floats of length 3. The xyz position of the atom.
        element: int or str. The element of the atom.
        q: float. The charge of the atom.
        classes: list of lists of int or str. Chemical elements for atom classification. Can be atomic numbers or letters.
        class_weights: list of float that sum to one. List of weights or one-hot vector for classes.
    Note: only one of classes and class_weights can be specified at the same time.
    '''
    def __init__(self, xyz, element=None, q=None, classes=None, class_weights=None):

        self.xyz = list(xyz)

        if element is not None:
            if isinstance(element, str):
                try:
                    element = elements.index(element) + 1
                except ValueError:
                    raise ValueError(f'Invalid element {element} for atom.')
            self.element = element
        else:
            self.element = None

        if q is None:
            q = 0
        self.q = 0

        if classes is not None:
            assert class_weights is None, 'Cannot have both classes and class_weights not be None.'
            self.class_weights, self.class_index = self._get_class(classes)
        elif class_weights is not None:
            assert np.allclose(sum(class_weights), 1.0), "Class weights don't sum to unity."
            self.class_weights = list(class_weights)
            self.class_index = np.argmax(class_weights)
        else:
            self.class_weights = []
            self.class_index = None

    def _get_class(self, classes):
        cls_assign = [self.element in c for c in classes]
        try:
            ind = cls_assign.index(1)
        except ValueError:
            raise ValueError(f'Element {self.element} is not in any of the classes.')
        return list(np.eye(len(classes))[ind]), ind
    
    def array(self, xyz=False, q=False, element=False, class_index=False, class_weights=False):
        '''
        Return an array representation of the atom in order [xyz, q, element, class_index, one_hot_class]
        Arguments:
            xyz: Bool. Include xyz coordinates.
            q: Bool. Include charge.
            element: Bool. Include element.
            class_index: Bool. Include class index.
            class_weights: Bool. Include class weights.
        Returns: np.ndarray with ndim 1.
        '''
        arr = []
        if xyz:
            arr += self.xyz
        if q:
            arr += [self.q]
        if element:
            arr += [self.element]
        if class_index:
            arr += [self.class_index]
        if class_weights:
            arr += self.class_weights
        return np.array(arr) 

class MoleculeGraph:
    '''
    A class representing a molecule graph with atoms and bonds. The atoms are stored as a list of
    Atom objects.
    
    Arguments: 
        atoms: list of Atom or np.array of shape (num_atoms, 4). Molecule atom position and elements.
               Each row corresponds to one atom with [x, y, z, element].
        bonds: list of tuples (bond_start, bond_end). Indices of bonded atoms.
        classes: list of lists of int or str. Chemical elements for atom classification. Can be atomic numbers or letters.
        class_weights: list of lists of float that sum to one. List of weights or one-hot vector for classes of each atom.
    Note: only one of classes and class_weights can be specified at the same time.
    '''

    def __init__(self, atoms, bonds, classes=None, class_weights=None):
        if class_weights is not None:
            assert len(atoms) == len(class_weights), "The number of atoms and the number of class weights for atoms don't match"
        else:
            class_weights = [None] * len(atoms)
        self.atoms = []
        for atom, cw in zip(atoms, class_weights):
            if isinstance(atom, Atom):
                self.atoms.append(atom)
            else:
                self.atoms.append(Atom(atom[:3], atom[-1], q=None, classes=classes, class_weights=cw))
        self.bonds = bonds

    def __len__(self):
        return len(self.atoms)

    def remove_atoms(self, remove_inds):
        '''
        Remove atoms and corresponding bonds from a molecule graph.
        
        Arguments:
            remove_inds: list of int. Indices of atoms to remove.
        
        Returns:
            new_molecule: MoleculeGraph. New molecule graph where the atoms and bonds have been removed.
            removed: list of tuples (atom, bonds) where atom is a np.ndarray of length 4 and bonds is a list with length
                     equal to the number of remaining atoms in the molecule. Removed atoms and bonds.
                     Each list item corresponds to one of the removed atoms. The bonds are encoded as an indicator list
                     where 0 indicates no bond and 1 indicates a bond with the atom at the corresponding index in the new molecule.
        '''

        remove_inds = np.array(remove_inds, dtype=int)
        assert (remove_inds < len(self.atoms)).all()

        # Remove atoms from molecule
        removed_atoms = [self.atoms[i] for i in remove_inds]
        new_atoms = [self.atoms[i] for i in range(len(self.atoms)) if i not in remove_inds]

        # Remove corresponding bonds from molecule
        removed_bonds = [[0]*len(new_atoms) for _ in range(len(remove_inds))]
        new_bonds = []

        for bond in self.bonds:

            bond0 = bond[0] - (remove_inds < bond[0]).sum()
            bond1 = bond[1] - (remove_inds < bond[1]).sum()
            
            if not (bond[0] in remove_inds or bond[1] in remove_inds):
                new_bonds.append((bond0, bond1))

            elif not (bond[0] in remove_inds and bond[1] in remove_inds):
                for i in range(len(remove_inds)):
                    if bond[0] == remove_inds[i]:
                        removed_bonds[i][bond1] = 1
                    elif bond[1] == remove_inds[i]:
                        removed_bonds[i][bond0] = 1

        new_molecule = MoleculeGraph(new_atoms, new_bonds)
        removed = [(atom, bonds) for atom, bonds in zip(removed_atoms, removed_bonds)]

        return new_molecule, removed

    def add_atom(self, atom, bonds):
        '''
        Add an atom and bonds to molecule graph.
        
        Arguments:
            atom: Atom. Atom to add.
            bonds: list of 0s and 1s. Indicator list of bond connections from new atom to existing atoms in the graph.
        
        returns: MoleculeGraph. New molecule graph where the atom and bonds have been added.
        '''
        n_atoms = len(self.atoms)
        new_atoms = self.atoms + [atom]
        new_bonds = self.bonds + [(i, n_atoms) for i, b in enumerate(bonds) if b == 1]
        new_molecule = MoleculeGraph(new_atoms, new_bonds)
        return new_molecule

    def permute(self, permutation):
        '''
        Permute the index order of atoms and corresponding bond indices.
        
        Arguments:
            permutation: list of int. New index order. Has to be same length as number of atoms in graph.

        Returns: MoleculeGraph. New molecule graph with indices permuted.
        '''
        if len(permutation) != len(self.atoms):
            raise ValueError(f'Length of permutation list {len(permutation)} does not match '
                + f'the number of atoms in graph {len(self.atoms)}')
        new_atoms = [self.atoms[i] for i in permutation]
        new_bonds = []
        for b in self.bonds:
            new_bonds.append((permutation.index(b[0]), permutation.index(b[1])))
        new_molecule = MoleculeGraph(new_atoms, new_bonds)
        return new_molecule

    def array(self, xyz=False, q=False, element=False, class_index=False, class_weights=False):
        '''
        Return an array representation of the atoms in the molecule in order [xyz, q, element, class_index, class_weights]
        
        Arguments:
            xyz: Bool. Include xyz coordinates.
            q: Bool. Include charge.
            element: Bool. Include element.
            class_index: Bool. Include class index.
            class_weights: Bool. Include class weights.
        Returns: np.ndarray with ndim 2. Each element in first dimension corresponds to one atom.
        '''
        if len(self.atoms) > 0:
            arr = np.stack([atom.array(xyz, q, element, class_index, class_weights) for atom in self.atoms], axis=0)
        else:
            arr = []
        return arr

    def adjacency_matrix(self):
        '''
        Return the adjacency matrix of the graph.

        Returns: np.ndarray of shape (n_atoms, n_atoms), where n_atoms is the number of atoms in the graph.
            Adjacency matrix, where the presence of bonds between pairs of atoms are indicated by a binary values.
        '''
        A = np.zeros((len(self.atoms), len(self.atoms)), dtype=int)
        bonds = np.array(self.bonds, dtype=int).T
        if len(bonds) > 0:
            b0, b1 = bonds[0], bonds[1]
            np.add.at(A, (b0, b1), 1)
            np.add.at(A, (b1, b0), 1)
        return A

    def transform_xy(self, rot_xy=0, flip_x=False, flip_y=False, center=(0, 0)):
        '''
        Transform atom positions in the xy plane.

        Arguments:
            rot_xy: float. Rotate atoms in xy plane by rot_xy degrees around center point.
            flip_x: bool. Mirror atom positions in x direction with respect to the center point.
            flip_y: bool. Mirror atom positions in y direction with respect to the center point.
            center: tuple (x, y). Point around which transformations are performed.

        Returns: MoleculeGraph. A new molecule graph with rotated atom positions.
        '''

        center = np.array(center)
        atom_pos = self.array(xyz=True)

        if rot_xy:
            a = rot_xy / 180 * np.pi
            rot_mat = np.array([
                [np.cos(a), -np.sin(a)],
                [np.sin(a),  np.cos(a)]
            ])
            atom_pos[:, :2] -= center
            atom_pos[:, :2] = np.dot(atom_pos[:, :2], rot_mat.T)
            atom_pos[:, :2] += center

        if flip_x:
            atom_pos[:, 0] = 2*center[0] - atom_pos[:, 0]
        
        if flip_y:
            atom_pos[:, 1] = 2*center[1] - atom_pos[:, 1]

        new_atoms = []
        for atom, pos in zip(self.atoms, atom_pos):
            new_atom = copy.deepcopy(atom)
            new_atom.xyz = list(pos)
            new_atoms.append(new_atom)
        new_bonds = copy.deepcopy(self.bonds)

        return MoleculeGraph(new_atoms, new_bonds)

    def crop_atoms(self, box_borders):
        '''
        Delete atoms that are outside of specified region.
        
        Arguments:
            box_borders: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent
                of the region outside of which atoms are deleted.

        Returns: MoleculeGraph. A new molecule graph without the deleted atoms.
        '''

        remove_inds = []
        for i, atom in enumerate(self.atoms):
            pos = atom.array(xyz=True)
            if not (
                box_borders[0][0] <= pos[0] <= box_borders[1][0] and
                box_borders[0][1] <= pos[1] <= box_borders[1][1] and
                box_borders[0][2] <= pos[2] <= box_borders[1][2]
                ):
                remove_inds.append(i)
        
        new_molecule, _ = self.remove_atoms(remove_inds)

        return new_molecule

def find_bonds(molecules, tolerance=0.2):
    '''
    Arguments:
        molecules: list of np.array of shape (num_atoms, 4). Molecule atom position and elements.
                   Each row corresponds to one atom with [x, y, z, element].
        tolerance: float. Two atoms are bonded if their distance is at most by a factor of
                   1+tolerance as long as the table value for the bond.
    
    Returns: list of lists of tuples (bond_start, bond_end). Indices of bonded atoms.
    '''
    bonds = []
    for mol in molecules:
        bond_ind = []
        for i in range(len(mol)):
            for j in range(len(mol)):
                if j <= i: continue
                atom_i = mol[i]
                atom_j = mol[j]
                r = np.linalg.norm(atom_i[:3] - atom_j[:3])
                elems = sorted([atom_i[-1], atom_j[-1]])
                bond_length = bond_lengths[elems[0]][elems[1]]
                if r < (1+tolerance)*bond_length:
                    bond_ind.append((i,j))
        bonds.append(bond_ind)
    return bonds
    
def threshold_atoms_bonds(molecules, threshold=-1.0, use_vdW=False):
    '''
    Remove atoms and corresponding bonds beyond threshold depth in molecules.
    
    Arguments: 
        molecules: list of MoleculeGraph. Molecules to threshold.
        threshold: float. Deepest z-coordinate for included atoms (top is 0).
        use_vdW: Boolean. Whether to add vdW radii to the atom z coordinates when calculating depth.
    
    Returns: 
        new_molecules: list of MoleculeGraph. Molecules with deep atoms removed.
    '''
    new_molecules = []
    for mol in molecules:
        zs = mol.array(xyz=True)[:,2].copy()
        if use_vdW:
            zs += np.fromiter(map(lambda i: vdW_radii[i], mol.array(element=True)[:,0]), dtype=np.float64)
        zs -= zs.max()
        remove_inds = np.where(zs < threshold)[0]
        new_molecule, removed = mol.remove_atoms(remove_inds)
        new_molecules.append(new_molecule)
    return new_molecules

def remove_single_atom(molecules):
    '''
    Randomly remove a single atom and corresponding bonds from molecules.
    
    Arguments: 
        molecules: list of MoleculeGraph. Molecules to modify.
    
    Returns: 
        new_molecules: list of MoleculeGraph. Molecules after removing atoms.
        removed: list of tuples (atom, bonds) where atom is a np.ndarray of length 4 and bonds is a list with length
                 equal to the number of remaining atoms in the molecule. Removed atoms and bonds.
                 Each list item corresponds to one molecule. The bonds are encoded as an indicator list where 0 indicates
                 no bond and 1 indicates a bond with the atom at the corresponding index in the new molecule.
    '''
    
    removed = []
    new_molecules = []

    for mol in molecules:
        remove_ind = np.random.randint(0, len(mol))
        m, r = mol.remove_atoms([remove_ind])
        removed.append(r[0])
        new_molecules.append(m)
    
    return new_molecules, removed

def remove_atoms(molecules, order=None):
    '''
    Randomly remove several (or possibly all) atoms and corresponding bonds from molecules.
    The number of removed atoms is uniform random in {0, ..., n_atoms}, where n_atoms is the
    total number of atoms in a given molecule.
    
    Arguments: 
        molecules: list of MoleculeGraph. Molecules to modify.
        order: 'x', 'y', 'z', or None. Remove atoms in order based on increasing x, y, or z coordinate,
            or if None, then remove in random order.
    
    Returns: 
        removed: list of lists of tuples (atom, bonds) where atom is a np.ndarray of length 4 and bonds is a list with length
                 equal to the number of remaining atoms in the molecule. Removed atoms and bonds.
                 Each list item corresponds to one molecule and in second-level lists, each list item
                 corresponds to one of the removed atoms. The bonds are encoded as an indicator list where 0 indicates no bond
                 and 1 indicates a bond with the atom at the corresponding index in the new molecule.
        new_molecules: list of MoleculeGraph. Molecules after removing atoms.
    '''
    
    removed = []
    new_molecules = []

    for mol in molecules:
        n_remove = np.random.randint(0, len(mol)+1)
        if not order:
            remove_inds = np.random.choice(len(mol), size=n_remove, replace=False)
        elif order in ['x', 'y', 'z']:
            ind = 'xyz'.index(order)
            pos = mol.array(xyz=True)[:, ind]
            remove_inds = np.argsort(pos)[:n_remove]
        else:
            raise ValueError(f'Unknown removal order `{order}`.')
        m, r = mol.remove_atoms(remove_inds)
        removed.append(r)
        new_molecules.append(m)
        
    return new_molecules, removed

def randomize_atom_positions(atoms, std=[0.2, 0.2, 0.05], cutoff=0.5):
    '''
    Add random gaussian noise to atom positions.

    Arguments:
        atoms: list of Atom objects. Atoms whose positions to randomize.
        std: list of three floats. Standard deviation of added noise in angstroms.
        cutoff: float. Maximum displacement to position.
    '''
    for atom in atoms:
        if atom:
            disp = np.random.normal(0, std, 3)
            abs_disp = np.linalg.norm(disp)
            if abs_disp > cutoff:
                disp *= cutoff / abs_disp
            atom.xyz = [p + d for p, d in zip(atom.xyz, disp)]

def save_graphs_to_xyzs(molecules, classes, outfile_format='./{ind}_graph.xyz', start_ind=0, verbose=1):
    '''
    Save molecule graphs to xyz files.
    Arguments:
        molecules: list of MoleculeGraph. Molecule graphs to save.
        classes: list of lists of int or str. Chemical elements in each class. Can be atomic numbers or letters (e.g. 1 or 'H').
        outfile_format: str. Formatting string for saved files. Sample index is available in variable "ind".
        start_ind: int. Index where file numbering starts.
        verbose: int 0 or 1. Whether to print output information.
    '''

    ind = start_ind
    for mol in molecules:

        outfile = outfile_format.format(ind=ind)
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if len(mol) > 0:
            mol_xyz = mol.array(xyz=True)
            mol_elements = np.array([classes[int(m)][0] for m in mol.array(class_index=True).squeeze(1)])[:, None]
            mol_arr = np.append(mol_xyz, mol_elements, axis=1)
        else:
            mol_arr = np.empty((0,4))

        write_to_xyz(mol_arr, outfile, verbose=verbose)

        ind += 1

def make_position_distribution(atoms, box_borders, box_res=(0.125, 0.125, 0.1), std=0.3):
    '''
    Make a distribution on a grid based on atom positions. Each atom is represented by
    a normal distribution.

    Arguments:
        atoms: list of lists of Atom objects. The outer list corresponds to a batch item
            and the inner list to a single atom within the batch item.
        std: float. Standard deviation of normal distribution for each atom in angstroms.
        box_borders: tuple ((x_start, y_start, z_start),(x_end, y_end, z_end)). Real-space extent of the grid
            in angstroms.
        box_res: tuple (x_res, y_res, z_res). Real-space size of each voxel in angstroms.

    Returns: np.ndarray of size (n_batch, n_x, n_y, n_z).
    '''

    cov = std**2

    # Initialize grid and distribution
    n_batch = len(atoms)
    n_xyz = [int( (box_borders[1][i] - box_borders[0][i]) / box_res[i] ) for i in range(3)]
    xyz_start = [box_borders[0][i] + box_res[i] / 2 for i in range(3)]
    x, y, z = [np.arange(xyz_start[i], xyz_start[i] + n_xyz[i] * box_res[i], box_res[i])
        for i in range(3)]
    pos_grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    pos_dist = np.zeros([n_batch, n_xyz[0], n_xyz[1], n_xyz[2]])

    # Add atoms to the distribution
    for i in range(n_batch):
        for atom in atoms[i]:
            pos_dist[i] += multivariate_normal.pdf(pos_grid, mean=atom.array(xyz=True), cov=cov)

    return pos_dist

def _find_peaks_cpu(pos_dist, box_borders, match_threshold, std, method):

    n_xyz = pos_dist.shape[1:]
    res = [(box_borders[1][i] - box_borders[0][i]) / n_xyz[i] for i in range(3)]
    pos_dist[pos_dist < 1e-4] = 0 # Very small values cause instabilities in ZNCC values
    xyz_start = [box_borders[0][i] + res[i]/2 for i in range(3)]

    # Create reference gaussian peak to compare against
    r = 3 * std + 1e-6
    r = [r - (r % res[i]) for i in range(3)]
    x_ref, y_ref, z_ref = [np.arange(-r[i], r[i]+1e-6, res[i]) for i in range(3)]
    ref_grid = np.stack(np.meshgrid(x_ref, y_ref, z_ref, indexing='ij'), axis=-1)
    ref_peak = multivariate_normal.pdf(ref_grid, mean=[0, 0, 0], cov=std**2)

    # Match the reference gaussian peak shape with the position distributions
    if method in ['mad', 'msd', 'mad_norm', 'msd_norm']:
        matches = match_template_pool(pos_dist, ref_peak, method=method)
    else:
        matches = []
        for d in pos_dist:
            matches.append(
                feature.match_template(d, ref_peak, pad_input=True, mode='constant', constant_values=0)
            )
        matches = np.stack(matches, axis=0)

    # Threshold the match map
    if method == 'zncc':
        threshold_masks = matches > match_threshold
    else:
        threshold_masks = matches < match_threshold

    # Loop over batch items to label matches and find atom positions
    xyzs = []
    labels = []
    for match, threshold_mask in zip(matches, threshold_masks):
        
        # Label connected regions
        labels_, num_atoms = measure.label(threshold_mask, return_num=True)

        # Loop over labelled regions to find atom positions
        xyzs_ = []
        for target_label in range(1, num_atoms+1):

            # Find best matching xyz position from the labelled region
            match_masked = np.ma.array(match, mask=(labels_ != target_label))
            best_ind = match_masked.argmax() if method == 'zncc' else match_masked.argmin()
            best_ind = np.unravel_index(best_ind, match_masked.shape)
            xyz = [xyz_start[i] + res[i] * best_ind[i] for i in range(3)]

            xyzs_.append(xyz)

        xyzs.append(np.array(xyzs_))
        labels.append(labels_)
    
    labels = np.stack(labels, axis=0)

    return xyzs, matches, labels

def _find_peaks_cuda(pos_dist, box_borders, match_threshold, std, method):

    if method == 'zncc':
        raise NotImplementedError('zncc not implemented for cuda tensors.')

    n_xyz = pos_dist.shape[1:]
    res = [(box_borders[1][i] - box_borders[0][i]) / n_xyz[i] for i in range(3)]
    xyz_start = [box_borders[0][i] + res[i]/2 for i in range(3)]

    # Create reference gaussian peak to compare against
    r = 3 * std + 1e-6
    r = [r - (r % res[i]) for i in range(3)]
    x_ref, y_ref, z_ref = [np.arange(-r[i], r[i]+1e-6, res[i]) for i in range(3)]
    ref_grid = np.stack(np.meshgrid(x_ref, y_ref, z_ref, indexing='ij'), axis=-1)
    ref_peak = multivariate_normal.pdf(ref_grid, mean=[0, 0, 0], cov=std**2)
    ref_peak = torch.from_numpy(ref_peak).to(pos_dist.device).float()

    # Match the reference gaussian peak shape with the position distributions
    matches = match_template_cuda(pos_dist, ref_peak, method=method)

    # Threshold the match map
    threshold_masks = matches < match_threshold

    # Label matched regions
    labels = ccl(threshold_masks)

    # Find minimum indices in labelled regions
    min_inds = find_label_min(matches, labels)

    # Convert indices into real-space coordinates
    xyz_start = torch.tensor(xyz_start, device=pos_dist.device)
    res = torch.tensor(res, device=pos_dist.device)
    xyzs = [xyz_start + res * m for m in min_inds]

    return xyzs, matches, labels

def find_gaussian_peaks(pos_dist, box_borders, match_threshold=0.7, std=0.3, method='mad'):
    '''
    Find real-space positions of gaussian peaks in a 3D position distribution grid.

    Arguments:
        pos_dist: np.ndarray of torch.Tensor of shape (n_batch, n_x, n_y, n_z). Position distribution.
        box_borders: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent of the
            distribution grid in angstroms.
        match_threshold: float. Detection threshold for matching. Regions above the threshold are chosen for
            method 'zncc', and regions below the threshold are chosen for methods 'mad', 'msd, 'mad_norm', and
            'msd_norm'.
        std: float. Standard deviation of peaks to search for in angstroms.
        method: 'zncc', 'mad', 'msd', 'mad_norm', or 'msd_norm. Matching method to use. Either zero-normalized
            cross correlation ('zncc'), mean absolute distance ('mad'), mean squared distance ('msd'), or the
            normalized version of the latter two ('mad_norm', 'msd_norm').

    Returns: xyzs, match, labels
        xyzs: list of np.ndarray or torch.Tensor of shape (num_atoms, 3). Positions of the found atoms.
            Each item in theclist corresponds one batch item.
        matches: np.ndarray or torch.Tensor of same shape as input pos_dist. Array of matching values.
            For method 'zncc' larger values, and for 'mad', 'msd', 'mad_norm', and 'msd_norm' smaller
            values correspond to better match.
        labels: np.ndarray or torch.Tensor of same shape as input pos_dist. Labelled regions where
            match is better than match_threshold.
    '''

    if method not in ['zncc', 'mad', 'msd', 'mad_norm', 'msd_norm']:
        raise ValueError(f'Unknown matching method `{method}`.')

    if isinstance(pos_dist, torch.Tensor):
        if pos_dist.device == torch.device('cpu'):
            xyzs, matches, labels = _find_peaks_cpu(pos_dist.numpy(), box_borders,
                match_threshold, std, method)
            xyzs = [torch.from_numpy(xyz).type(pos_dist.dtype) for xyz in xyzs]
            matches = torch.from_numpy(matches).type(pos_dist.dtype)
            labels = torch.from_numpy(labels).type(pos_dist.dtype)
        else:
            xyzs, matches, labels = _find_peaks_cuda(pos_dist, box_borders,
                match_threshold, std, method)
    else:
        xyzs, matches, labels = _find_peaks_cpu(pos_dist, box_borders,
            match_threshold, std, method)

    return xyzs, matches, labels

def download_molecules(save_path='./Molecules', verbose=1):
    '''
    Download database of molecules.
    
    Arguments:
        save_path: str. Path where the molecule xyz files will be saved.
        verbose: int 0 or 1. Whether to print progress information.
    '''
    if not os.path.exists(save_path):
        download_url = 'https://www.dropbox.com/s/z4113upq82puzht/Molecules_rebias_210611.tar.gz?dl=1'
        temp_file = '.temp_molecule.tar'
        if verbose: print('Downloading molecule tar archive...')
        temp_file, info = urlretrieve(download_url, temp_file)
        if verbose: print('Extracting tar archive...')
        with tarfile.open(temp_file, 'r') as f:
            base_dir = os.path.normpath(f.getmembers()[0].name).split(os.sep)[0]
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f)
        if verbose: print('Done extracting.')
        shutil.move(base_dir, save_path)
        os.remove(temp_file)
        if verbose: print(f'Moved files to {save_path}.')
    else:
        if verbose: print(f'Target folder {save_path} already exists. Skipping downloading molecules.')

def read_xyzs(file_paths, return_comment=False):
    '''
    Read molecule xyz files.
    Arguments:
        file_paths: list of str. Paths to xyz files
        return_comment: bool. If True, also return the comment string on second line of file.
    Returns: list of np.array of shape (num_atoms, 4) or (num_atoms, 5). Each row
             corresponds to one atom with [x, y, z, element] or [x, y, z, charge, element].
    '''
    mols = []
    comments = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            N = int(f.readline().strip())
            comments.append(f.readline())
            atoms = []
            for line in f:
                line = line.strip().split()
                try:
                    elem = int(line[0])
                except ValueError:
                    elem = elements.index(line[0]) + 1
                posc = [float(p) for p in line[1:]]
                atoms.append(posc + [elem])
        mols.append(np.array(atoms))
    if return_comment:
        mols = mols, comments
    return mols

def write_to_xyz(molecule, outfile='./pos.xyz', comment_str='', verbose=1):
    '''
    Write molecule into xyz file.

    Arguments:
        molecule: np.array of shape (num_atoms, 4) or (num_atoms, 5). Molecule to write.
                  Each row corresponds to one atom with [x, y, z, element] or [x, y, z, charge, element].
        outfile: str. Path where xyz file will be saved.
        comment_str: str. Comment written to the second line of the file.
        verbose: int 0 or 1. Whether to print output information.
    '''
    molecule = molecule[molecule[:,-1] > 0]
    with open(outfile, 'w') as f:
        f.write(f'{len(molecule)}\n{comment_str}\n')
        for atom in molecule:
            f.write(f'{int(atom[-1])}\t')
            for i in range(len(atom)-1):
                f.write(f'{atom[i]:10.8f}\t')
            f.write('\n')
    if verbose > 0: print(f'Molecule xyz file saved to {outfile}')
    
def batch_write_xyzs(xyzs, outdir='./', start_ind=0, verbose=1):
    '''
    Write a batch of xyz files 0_mol.xyz, 1_mol.xyz, ...

    Arguments:
        xyzs: list of np.array of shape (num_atoms, 4) or (num_atoms, 5). Molecules to write.
        outdir: str. Directory where files are saved.
        start_ind: int. Index where file numbering starts.
        verbose: int 0 or 1. Whether to print output information.
    '''
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    ind = start_ind
    for xyz in xyzs:
        write_to_xyz(xyz, os.path.join(outdir, f'{ind}_mol.xyz'), verbose=verbose)
        ind += 1

def count_parameters(module):
    '''
    Count pytorch module parameters.

    Arguments:
        module: torch.nn.Module.
    '''
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, save_dir, lr_scheduler=None):
    '''
    Save pytorch checkpoint.

    Arguments:
        model: torch.nn.Module.
        optimizer: torch.optim.Optimizer.
        epoch: int. Training epoch.
        save_dir: str. Directory to save in.
        lr_scheduler: torch.optim.lr_scheduler.
    '''
    import torch

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if hasattr(model, 'module'):
        model = model.module

    state = {
        'model_params': model.state_dict(),
        'optim_params': optimizer.state_dict(),
        }
    if lr_scheduler is not None:
        state['scheduler_params'] = lr_scheduler.state_dict()
        
    torch.save(state, os.path.join(save_dir, f'model_{epoch}.pth'))
    print(f'Model, optimizer weights on epoch {epoch} saved to {save_dir}')
        
def load_checkpoint(model, optimizer=None, file_name='./model.pth', lr_scheduler=None):
    '''
    Load pytorch checkpoint.

    Arguments:
        model: torch.nn.Module.
        optimzer: torch.optim.Optimizer.
        file_name: str. Checkpoint file to load from.
        lr_scheduler: torch.optim.lr_scheduler.
    '''
    import torch

    state = torch.load(file_name)
    model.load_state_dict(state['model_params'])

    if optimizer:
        optimizer.load_state_dict(state['optim_params'])
        msg = f'Model, optimizer weights loaded from {file_name}'
    else:
        msg = f'Model weights loaded from {file_name}'

    if lr_scheduler is not None:
        try:
        	lr_scheduler.load_state_dict(state['scheduler_params'])
        except:
            print('Learning rate scheduler parameters could not load.')
            
    print(msg)

class LossLogPlot:
    '''
    Log and plot model training loss history. Add epoch losses with add_losses and plot with plot_history.

    Arguments:
        log_path: str. Path where loss log is saved.
        plot_path: str. Path where plot of loss history is saved.
        loss_labels: list of str. Labels for different loss components.
        loss_weights: list of int or str. Weights for different loss components.
            Empty string for no weight (e.g. Total loss).
    '''
    def __init__(self, log_path, plot_path, loss_labels, loss_weights=None):
        self.log_path = log_path
        self.plot_path = plot_path
        self.loss_labels = loss_labels
        if not loss_weights:
            self.loss_weights = [''] * len(self.loss_labels)
        else:
            assert len(loss_weights) == len(loss_labels)
            self.loss_weights = loss_weights
        self.train_losses = np.empty((0, len(loss_labels)))
        self.val_losses = np.empty((0, len(loss_labels)))
        self.epoch = 0
        self._init_log()

    def _init_log(self):
        if not(os.path.isfile(self.log_path)):
            with open(self.log_path, 'w') as f:
                f.write('epoch')
                for i, label in enumerate(self.loss_labels):
                    label = f';train_{label}'
                    if self.loss_weights[i]:
                        label += f' (x {self.loss_weights[i]})'
                    f.write(label)
                for i, label in enumerate(self.loss_labels):
                    label = f';val_{label}'
                    if self.loss_weights[i]:
                        label += f' (x {self.loss_weights[i]})'
                    f.write(label)
                f.write('\n')
            print(f'Created log at {self.log_path}')
        else:
            with open(self.log_path, 'r') as f:
                header = f.readline().rstrip('\r\n').split(';') 
                hl = (len(header)-1) // 2
                if len(self.loss_labels) != hl:
                    raise ValueError(f'The length of the given list of loss names and the length of the header of the existing log at {self.log_path} do not match.')
                for line in f:
                    line = line.rstrip('\n').split(';')
                    self.train_losses = np.append(self.train_losses, [[float(s) for s in line[1:hl+1]]], axis=0)
                    self.val_losses = np.append(self.val_losses, [[float(s) for s in line[hl+1:]]], axis=0)
                    self.epoch += 1
            print(f'Using existing log at {self.log_path}')
    
    def add_losses(self, train_loss, val_loss):
        '''
        Add losses to log.

        Arguments:
            train_loss: list of floats of length len(self.loss_labels). Training losses for the epoch.
            val_loss: list of floats of length len(self.loss_labels). Validation losses for the epoch.
        '''
        self.epoch += 1
        self.train_losses = np.append(self.train_losses, [train_loss], axis=0)
        self.val_losses = np.append(self.val_losses, [val_loss], axis=0)
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch))
            for l in train_loss:
                f.write(f';{l}')
            for l in val_loss:
                f.write(f';{l}')
            f.write('\n')
        
    def plot_history(self, show=False):
        '''
        Plot and save history of current losses into self.plot_path.

        Arguments:
            show: Bool. Whether to show the plot on screen.
        '''
        x = range(1, self.epoch+1)
        n_rows, n_cols = _calc_plot_dim(len(self.loss_labels), f=0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, (label, ax) in enumerate(zip(self.loss_labels, axes.flatten())):
            ax.semilogy(x, self.train_losses[:,i],'-bx')
            ax.semilogy(x, self.val_losses[:,i],'-gx')
            ax.legend(['Training', 'Validation'])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            if self.loss_weights[i]:
                label = f'{label} (x {self.loss_weights[i]})'
            ax.set_title(label)
        fig.tight_layout()
        plt.savefig(self.plot_path)
        print(f'Loss history plot saved to {self.plot_path}')
        if show:
            plt.show()
        else:
            plt.close()

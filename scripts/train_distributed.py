
import os
import sys
import time
import copy
import random
import numpy as np

import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append('../src') # Add source code directory to Python PATH
import utils
import analysis
import preprocessing as pp
import data_loading  as dl
import visualization as vis
from models import AttentionEncoderUNet, GNN, GridGraphImgNet, GridMultiAtomLoss

hdf5_path = './graph_dataset.hdf5'                       # Path to HDF5 database where data is read from
model_dir = './model_random_order'                       # Directory where all model files are saved
epochs = 50                                              # Number of epochs to train
batch_size = 32                                          # Number of samples per batch
classes = [[1], [6, 14], [7, 15], [8, 16], [9, 17, 35]]  # List of elements in each class
box_borders = ((2,2,-1.5),(18,18,0.5))                   # Real-space extent of plotting region
box_res = (0.125, 0.125, 0.1)                            # Real-space voxel size for position distribution
zmin = -0.8                                              # Maximum depth used for thresholding
peak_std = 0.25                                          # Standard deviation of atom position peaks in angstroms
sequence_order = None                                    # Order for graph construction
num_workers = 4                                          # Number of parallel workers for each GPU
timings = False                                          # Print timings for each batch
device = 'cuda'                                          # Device to use
comm_backend = 'nccl'                                    # Backend for interprocess communications
print_interval = 10                                      # Losses will be printed every print_interval batches
class_colors = ['w', 'dimgray', 'b', 'r', 'yellowgreen'] # Colors for classes
loss_weights = {                                         # Weights for loss components
    'pos_factor'        : 100.0,
    'class_factor'      : 1.0,
    'edge_factor'       : 1.0
}

def make_model(device):
    gnn = GNN(
        hidden_size     = 64,
        iters           = 3,
        n_node_features = 20,
        n_edge_features = 20
    )
    cnn = AttentionEncoderUNet(
        conv3d_in_channels      = 1,
        conv3d_block_channels   = [4, 8, 16, 32],
        conv3d_block_depth      = 2,
        encoding_block_channels = [4, 8, 16, 32],
        encoding_block_depth    = 2,
        upscale_block_channels  = [32, 16, 8],
        upscale_block_depth     = 2,
        upscale_block_channels2 = [32, 16, 8],
        upscale_block_depth2    = 2,
        attention_channels      = [32, 32, 32],
        query_size              = 64,
        res_connections         = True,
        hidden_dense_units      = [],
        out_units               = 128,
        activation              = 'relu',
        padding_mode            = 'zeros',
        pool_type               = 'avg',
        pool_z_strides          = [2, 1, 2],
        decoder_z_sizes         = [4, 10, 20],
        attention_activation    = 'softmax'
    )
    model = GridGraphImgNet(
        cnn                  = cnn,
        gnn                  = gnn,
        n_classes            = len(classes),
        expansion_hidden     = 32,
        expanded_size        = 128,
        query_hidden         = 64,
        class_hidden         = 32,
        edge_hidden          = 32,
        peak_std             = peak_std,
        match_method         = 'msd_norm',
        match_threshold      = 0.7,
        dist_threshold       = 0.5,
        teacher_forcing_rate = 1.0,
        device               = device
    )
    criterion = GridMultiAtomLoss(**loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, criterion, optimizer

def apply_preprocessing(batch):

    X, Y, atoms = batch
    
    pp.add_norm(X)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.rand_shift_xy_trend(X, shift_step_max=0.02, max_shift_total=0.04)
    pp.add_cutout(X, n_holes=5)
    X = X[0]
    
    atoms = pp.top_atom_to_zero(atoms)
    xyz = atoms.copy()
    bonds = utils.find_bonds(atoms)
    mols = [utils.MoleculeGraph(a, b, classes=classes) for a, b in zip(atoms, bonds)]
    mols = utils.threshold_atoms_bonds(mols, zmin)
    ref_graphs = copy.deepcopy(mols)
    mols, removed = utils.remove_atoms(mols, order=sequence_order)

    ref_dist = utils.make_position_distribution([m.atoms for m in ref_graphs], box_borders,
        box_res=box_res, std=peak_std)
    
    if sequence_order:
        order_ind = 'xyz'.index(sequence_order)
        ref_atoms = [sorted(r, key=lambda x: x[0].xyz[order_ind])[-1][0] if len(r) > 0 else None
            for r in removed]
    else:
        ref_atoms = [random.choice(r)[0] if len(r) > 0 else None for r in removed]
    utils.randomize_atom_positions(ref_atoms, std=[0.2, 0.2, 0.05], cutoff=0.5)
    
    return X, mols, removed, xyz, ref_graphs, ref_dist, ref_atoms, box_borders

def make_dataloader(mode, world_size, rank):
    dataset, dataloader, sampler = dl.make_hdf5_dataloader(
        hdf5_path,
        apply_preprocessing,
        mode=mode,
        collate_fn=dl.collate,
        batch_size=batch_size // world_size,
        shuffle=False,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank
    )
    return dataset, dataloader, sampler

def batch_to_device(batch, device):
    X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs, box_borders = batch
    X = X.to(device)
    node_inputs = node_inputs.to(device)
    edges = edges.to(device)
    terminate = terminate.to(device)
    ref_dist = ref_dist.to(device)
    node_rem = [n.to(device) for n in node_rem]
    edge_rem = [e.to(device) for e in edge_rem]
    ref_atoms = ref_atoms.to(device)
    return X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs, box_borders

def loss_str(losses):
    loss_msg = (
        f'{losses[0]:.4f} ('
        f'MSE (Pos.): {losses[1]:.4f} x {loss_weights["pos_factor"]}, '
        f'NLL (Class): {losses[2]:.4f} x {loss_weights["class_factor"]}, '
        f'NLL (Edge): {losses[3]:.4f} x {loss_weights["edge_factor"]})'
    )
    return loss_msg

def recv_losses(losses, world_size, rank=0):
    losses_ = torch.zeros(world_size, len(losses)).float().to(rank)
    losses_[0] = torch.tensor(losses).float().to(rank)
    for i in range(1, world_size):
        dist.recv(losses_[i], src=i)
    losses = losses_.mean(dim=0)
    return losses

def run(rank, world_size, q):

    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(comm_backend, rank=rank, world_size=world_size)

    start_time = time.time()

    # Create directories
    checkpoint_dir = os.path.join(model_dir, 'CheckPoints/')
    if rank == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    # Define model, optimizer, and loss
    model, criterion, optimizer = make_model(rank)
    
    if rank == 0:
        print(f'CUDA is available = {torch.cuda.is_available()}')
        print(f'Model total parameters: {utils.count_parameters(model)}')

    # Create datasets and dataloaders
    train_set, train_loader, sampler = make_dataloader('train', world_size, rank)
    val_set, val_loader, _ = make_dataloader('val', world_size, rank)

    # Load checkpoint if available
    dist.barrier()
    for init_epoch in reversed(range(1, epochs+1)):
        if os.path.exists( state_file := os.path.join(checkpoint_dir, f'model_{init_epoch}.pth') ):
            state = torch.load(state_file, map_location={'cuda:0': f'cuda:{rank}'})
            model.load_state_dict(state['model_params'])
            optimizer.load_state_dict(state['optim_params'])
            init_epoch += 1
            break

    # Wrap model in DistributedDataParallel.
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # Setup logging
    if rank == 0:
        log_path = os.path.join(model_dir, 'loss_log.csv')
        plot_path = os.path.join(model_dir, 'loss_history.png')
        loss_labels = ['Total', 'MSE (Pos.)', 'NLL (Class)', 'NLL (Edge)']
        logger = utils.LossLogPlot(log_path, plot_path, loss_labels,
            ['', loss_weights['pos_factor'], loss_weights['class_factor'], loss_weights['edge_factor']])
    
    if rank == 0:
        if init_epoch <= epochs:
            print(f'\n ========= Starting training from epoch {init_epoch}')
        else:
            print('Model already trained')
    
    for epoch in range(init_epoch, epochs+1):

        if rank == 0: print(f'\n === Epoch {epoch}')

        if sampler:
            sampler.set_epoch(epoch)

        # Train
        if rank == 0:
            train_losses = []
            epoch_start = time.time()
            if timings: t0 = epoch_start

        model.train()
        for ib, batch in enumerate(train_loader):
            
            # Transfer batch to device
            (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms,
                Ns, xyz, ref_graphs, box_borders) = batch_to_device(batch, rank)

            if timings and rank == 0:
                torch.cuda.synchronize()
                t1 = time.time()
            
            # Forward
            pred = model(X, node_inputs, edges, Ns, ref_atoms, box_borders)
            losses, _ = criterion(pred, (node_rem, edge_rem, terminate, ref_dist), separate_loss_factors=True)
            loss = losses[0]
            
            if timings and rank == 0: 
                torch.cuda.synchronize()
                t2 = time.time()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                
                losses = recv_losses(losses, world_size) # Get losses from other ranks
                train_losses.append([loss.item() for loss in losses])

                if ib == len(train_loader)-1 or (ib+1) % print_interval == 0:
                    eta = (time.time() - epoch_start) / (ib + 1) * ((len(train_loader)+len(val_loader)) - (ib + 1))
                    mean_loss = np.mean(train_losses[-print_interval:], axis=0)
                    loss_msg = f'Loss: {loss_str(mean_loss)}'
                    print(f'Epoch {epoch}, Train Batch {ib+1}/{len(train_loader)} - {loss_msg} - ETA: {eta:.2f}s')
                
                if timings:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    print(f'(Train) t0/Load Batch/Forward/Backward: {t0}/{t1-t0}/{t2-t1}/{t3-t2}')
                    t0 = t3
            else:

                # Send loss to rank 0
                dist.send(torch.tensor(losses).float().to(rank), dst=0)

        # Validate
        if rank == 0:
            val_losses = []
            val_start = time.time()
            if timings: t0 = val_start
        
        model.eval()
        with torch.no_grad():
            
            for ib, batch in enumerate(val_loader):
                
                # Transfer batch to device
                (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms,
                    Ns, xyz, ref_graphs, box_borders) = batch_to_device(batch, rank)
                
                if timings: 
                    torch.cuda.synchronize()
                    t1 = time.time()
                
                # Forward
                pred = model(X, node_inputs, edges, Ns, ref_atoms, box_borders)
                losses, _ = criterion(pred, (node_rem, edge_rem, terminate, ref_dist), separate_loss_factors=True)
                
                if rank == 0:

                    losses = recv_losses(losses, world_size) # Get losses from other ranks
                    val_losses.append([loss.item() for loss in losses])

                    if (ib+1) % print_interval == 0:
                        eta = (time.time() - epoch_start) / (len(train_loader) + ib + 1) * (len(val_loader) - (ib + 1))
                        print(f'Epoch {epoch}, Val Batch {ib+1}/{len(val_loader)} - ETA: {eta:.2f}s')
                    
                    if timings:
                        torch.cuda.synchronize()
                        t2 = time.time()
                        print(f'(Val) t0/Load Batch/Forward: {t0}/{t1-t0}/{t2-t1}')
                        t0 = t2

                else:

                    # Send loss to rank 0
                    dist.send(torch.tensor(losses).float().to(rank), dst=0)

        if rank == 0:

            train_loss = np.mean(train_losses, axis=0)
            val_loss = np.mean(val_losses, axis=0)

            print(f'End of epoch {epoch}')
            print(f'Train loss: {loss_str(train_loss)}')
            print(f'Val loss: {loss_str(val_loss)}')

            epoch_end = time.time()
            train_step = (val_start - epoch_start) / len(train_loader)
            val_step = (epoch_end - val_start) / len(val_loader)
            print(f'Epoch time: {epoch_end - epoch_start:.2f}s - Train step: {train_step:.5f}s '
                f'- Val step: {val_step:.5f}s')
        
            # Add losses to log
            logger.add_losses(train_loss, val_loss)
            logger.plot_history()
            
            # Save checkpoint
            utils.save_checkpoint(model, optimizer, epoch, checkpoint_dir)
        
    # Save final model
    if rank == 0:
        torch.save(model.module, save_path := os.path.join(model_dir, 'model.pth'))
        print(f'\nModel saved to {save_path}')

    print(f'Done at rank {rank}. Total time: {time.time() - start_time:.0f}s')

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(maxsize=10)
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, q), nprocs=world_size, join=True)
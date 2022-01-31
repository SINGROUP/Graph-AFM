
from train import *

# Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Inference device
device = 'cuda'

# How many test set batches to make predictions on
pred_batches = 2

def batch_to_host(pred, batch):
    X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs = batch
    pred_nodes, pred_edges, pred_dist = pred
    X = X.cpu()
    node_inputs = node_inputs.cpu()
    edges = edges.cpu()
    node_rem = [n.cpu() for n in node_rem]
    edge_rem = [e.cpu() for e in edge_rem]
    terminate = terminate.cpu()
    ref_dist = ref_dist.cpu()
    ref_atoms = ref_atoms.cpu()
    pred_nodes = pred_nodes.cpu()
    pred_dist = pred_dist.cpu()
    pred_edges = [e.cpu() for e in pred_edges]
    X, mols, pred, ref, xyz, ref_dist, pred_dist, ref_graphs, ref_atoms = dl.uncollate(
        (pred_nodes, pred_edges, pred_dist),
        (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs)
    )
    return X, mols, pred, ref, xyz, ref_dist, pred_dist, ref_graphs, ref_atoms

if __name__ == "__main__":

    start_time = time.time()

    # Check checkpoint directory
    checkpoint_dir = os.path.join(model_dir, 'CheckPoints/')
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError('No checkpoint directory. Cannot load model for testing.')
    
    # Define model, optimizer, and loss
    model, criterion, optimizer = make_model(device)
    
    print(f'CUDA is AVAILABLE = {torch.cuda.is_available()}')
    print(f'Model total parameters: {utils.count_parameters(model)}')

    # Create dataset and dataloader
    test_set, test_loader, _ = make_dataloader('test', world_size=1, rank=device)

    # Load checkpoint
    for last_epoch in reversed(range(1, epochs+1)):
        if os.path.exists( state_file := os.path.join(checkpoint_dir, f'model_{last_epoch}.pth') ):
            state = torch.load(state_file, map_location={'cuda:0': device})
            model.load_state_dict(state['model_params'])
            optimizer.load_state_dict(state['optim_params'])
            break
    
    print(f'\n ========= Testing with model from epoch {last_epoch}')

    stats = analysis.GraphPredStats(len(classes))
    seq_stats = analysis.GraphSeqStats(len(classes))
    eval_losses = []
    eval_start = time.time()
    if timings: t0 = eval_start
    
    model.eval()
    with torch.no_grad():
        
        for ib, batch in enumerate(test_loader):
                
            # Transfer batch to device
            (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms,
                Ns, xyz, ref_graphs, box_borders) = batch_to_device(batch, device)
            
            if timings:
                torch.cuda.synchronize()
                t1 = time.time()
            
            # Forward
            pred = model(X, node_inputs, edges, Ns, ref_atoms, box_borders, return_attention=True)
            pred_nodes, pred_edges, pred_dist, unet_attention_maps, encoding_attention_maps = pred
            losses, min_inds = criterion(
                (pred_nodes, pred_edges, pred_dist),
                (node_rem, edge_rem, terminate, ref_dist),
                separate_loss_factors=True
            )
                
            if timings:
                torch.cuda.synchronize()
                t2 = time.time()

            # Predict full molecule graphs in sequence
            pred_graphs, pred_dist, pred_sequence, completed = model.predict_sequence(X, box_borders,
                order=sequence_order)

            if timings:
                torch.cuda.synchronize()
                t3 = time.time()

            # Back to host
            batch = (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs)
            X, mols, pred, ref, xyz, ref_dist, pred_dist, ref_graphs, ref_atoms = batch_to_host(
                (pred_nodes, pred_edges, pred_dist), batch)

            # Gather statistical information
            ref = [r[i] + (t, d) for i, (r, t, d) in zip(min_inds, ref)]
            stats.add_batch_grid(pred, ref)
            seq_stats.add_batch(pred_graphs, ref_graphs)
            eval_losses.append([loss.item() for loss in losses])

            if (ib+1) % print_interval == 0:
                eta = (time.time() - eval_start) / (ib + 1) * (len(test_loader) - (ib + 1))
                print(f'Test Batch {ib+1}/{len(test_loader)} - ETA: {eta:.2f}s')
            
            if timings:
                torch.cuda.synchronize()
                t4 = time.time()
                print('(Test) t0/Load Batch/Forward/Seq prediction/Stats: '
                    f'{t0:6f}/{t1-t0:6f}/{t2-t1:6f}/{t3-t2:6f}/{t4-t3:6f}')
                t0 = t4

    # Save statistical information
    stats_dir1 = os.path.join(model_dir,'stats_single')
    stats_dir2 = os.path.join(model_dir,'stats_sequence')
    stats.plot(stats_dir1)
    stats.report(stats_dir1)
    seq_stats.plot(stats_dir2)
    seq_stats.report(stats_dir2)

    # Average losses and print
    eval_loss = np.mean(eval_losses, axis=0)
    print(f'Test set loss: {loss_str(eval_loss)}')

    # Save test set loss to file
    with open(os.path.join(model_dir, 'test_loss.txt'),'w') as f:
        f.write(';'.join([str(l) for l in eval_loss]))
  
    # Make predictions
    print(f'\n ========= Predict on {pred_batches} batches from the test set')
    counter = 0
    pred_dir = os.path.join(model_dir, 'predictions/')
    pred_dir2 = os.path.join(model_dir, 'predictions_sequence/')
    
    with torch.no_grad():
        
        for ib, batch in enumerate(test_loader):
        
            if ib >= pred_batches: break
            
            # Transfer batch to device
            (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms,
                Ns, xyz, ref_graphs, box_borders) = batch_to_device(batch, device)
            
            # Forward
            pred = model(X, node_inputs, edges, Ns, ref_atoms, box_borders, return_attention=True)
            pred_nodes, pred_edges, pred_dist, unet_attention_maps, encoding_attention_maps = pred
            _, min_inds = criterion(
                (pred_nodes, pred_edges, pred_dist),
                (node_rem, edge_rem, terminate, ref_dist),
                separate_loss_factors=True
            )
                
            # Predict full molecule graphs in sequence
            pred_graphs, pred_dist, pred_sequence, completed = model.predict_sequence(X, box_borders,
                order=sequence_order)

            # Back to host
            batch = (X, node_inputs, edges, node_rem, edge_rem, terminate, ref_dist, ref_atoms, Ns, xyz, ref_graphs)
            X, mols, pred, ref, xyz, ref_dist, pred_dist, ref_graphs, ref_atoms = batch_to_host(
                (pred_nodes, pred_edges, pred_dist), batch)
            unet_attention_maps = [a.cpu() for a in unet_attention_maps]
            encoding_attention_maps = [a.cpu() for a in encoding_attention_maps]

            # Save xyzs
            utils.batch_write_xyzs(xyz, outdir=pred_dir, start_ind=counter)
            utils.batch_write_xyzs(xyz, outdir=pred_dir2, start_ind=counter)
            utils.save_graphs_to_xyzs(pred_graphs, classes,
                outfile_format=os.path.join(pred_dir2, '{ind}_graph_pred.xyz'), start_ind=counter)
            utils.save_graphs_to_xyzs(ref_graphs, classes, 
                outfile_format=os.path.join(pred_dir2, '{ind}_graph_ref.xyz'), start_ind=counter)
        
            # Visualize predictions
            ref = [r[i] + (t, d) for i, (r, t, d) in zip(min_inds, ref)]
            vis.visualize_graph_grid(mols, pred, ref, box_borders=box_borders,
                outdir=pred_dir, start_ind=counter)
            vis.plot_graph_sequence_grid(pred_graphs, ref_graphs, pred_sequence, box_borders=box_borders,
                outdir=pred_dir2, start_ind=counter, classes=classes, class_colors=class_colors)
            vis.plot_distribution_grid(pred_dist, ref_dist, box_borders=box_borders, outdir=pred_dir2,
                start_ind=counter)
            vis.make_input_plots([X], outdir=pred_dir, start_ind=counter)
            vis.make_input_plots([X], outdir=pred_dir2, start_ind=counter)

            counter += len(mols)

    print(f'Done. Total time: {time.time() - start_time:.0f}s')

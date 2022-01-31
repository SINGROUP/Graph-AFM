
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff, cdist
from matplotlib.ticker import MaxNLocator

import visualization as vis

class GraphPredStats:
    '''
    Gather statistics on graph predictions of removed atoms.
    Arguments:
        n_classes: int. Number of classes.
        edge_conf: float in range [0,1]. Confidence threshold level for edge classification.
    '''
    def __init__(self, n_classes, edge_conf=0.5):
        self.n_classes = n_classes
        self.edge_conf = edge_conf
        self.conf_mat_node = np.zeros((n_classes, n_classes), dtype=np.int32)
        self.conf_mat_edge = np.zeros((2, 2), dtype=np.int32)
        self._ref_pos = []
        self._pred_pos = []
        self._pos_errors = []
        
    @property
    def ref_pos(self):
        return np.concatenate(self._ref_pos, axis=0)
    
    @property
    def pred_pos(self):
        return np.concatenate(self._pred_pos, axis=0)
        
    @property
    def pos_errors(self):
        return np.concatenate(self._pos_errors, axis=0)
        
    @property
    def pos_stats(self):
        pos_errors = self.pos_errors
        means = pos_errors.mean(axis=0)
        stds = pos_errors.std(axis=0)
        medians = np.median(pos_errors, axis=0)
        return means, medians, stds
        
    def pos_percentiles(self, percentiles):
        return np.percentile(self.pos_errors, percentiles, axis=0)
        
    def pos_depth_error(self, bins=20):
        pos_errors = self.pos_errors
        ref_z = self.ref_pos[:,2]
        if isinstance(bins, int):
            bins = np.linspace(ref_z.min(), ref_z.max()+1e-6, bins+1)
        inds = np.digitize(ref_z, bins)-1
        mean_errors = []
        for i in range(len(bins)-1):
            mean_errors.append(pos_errors[np.where(inds == i)].mean(axis=0))
        return np.array(mean_errors), bins
        
    @property
    def node_accuracy(self):
        return np.diag(self.conf_mat_node).sum() / self.conf_mat_node.sum()
        
    @property
    def node_precision(self):
        return np.diag(self.conf_mat_node) / self.conf_mat_node.sum(axis=0)
    
    @property
    def node_recall(self):
        return np.diag(self.conf_mat_node) / self.conf_mat_node.sum(axis=1)
    
    @property
    def edge_accuracy(self):
        return np.diag(self.conf_mat_edge).sum() / self.conf_mat_edge.sum()
    
    @property
    def edge_precision(self):
        return np.diag(self.conf_mat_edge) / self.conf_mat_edge.sum(axis=0)
    
    @property
    def edge_recall(self):
        return np.diag(self.conf_mat_edge) / self.conf_mat_edge.sum(axis=1)
        
    def add_batch(self, pred, ref):
        '''
        Gather stats from one batch of predictions and references.
        Arguments:
            pred: list of tuples (atom, bond), where atom/bond are np.ndarrays. Predictions.
            ref: list of tuples (atom, bond), where atom is np.ndarray and bond is a list. References.
        '''
        
        # Position error
        pred_pos = np.array([p[0][:3] for p in pred])
        ref_pos = np.array([r[0][:3] for r in ref])
        abs_err = np.abs(pred_pos - ref_pos)
        eucl_dist = np.sqrt((abs_err**2).sum(axis=1))
        self._pos_errors.append(np.append(eucl_dist[:, None], abs_err, axis=1)) 
        self._ref_pos.append(ref_pos[:,:3])
        self._pred_pos.append(pred_pos[:,:3])
        
        # Node classification confusion matrix
        pred_classes = np.argmax(np.array([p[0][3:] for p in pred]), axis=1)
        ref_classes = np.argmax(np.array([r[0][3:] for r in ref]), axis=1)
        np.add.at(self.conf_mat_node, (ref_classes, pred_classes), 1)
        
        # Edge classification confusion matrix
        pred_edges = np.concatenate([p[1] for p in pred])
        pred_edge_classes = np.zeros_like(pred_edges, dtype=np.int32)
        pred_edge_classes[pred_edges >= self.edge_conf] = 1
        ref_edge_classes = np.concatenate([r[1] for r in ref])
        np.add.at(self.conf_mat_edge, (ref_edge_classes, pred_edge_classes), 1)

    def add_batch_multiatom(self, pred, ref):
        '''
        Gather stats from one batch of predictions and references of multiatom model.
        Arguments:
            pred: List of tuples (atom, bonds), where atom is an Atom object and bonds is an indicator list.
                  Predicted atom position and bond connections.
            ref: list of lists tuple (atom, bonds), where atom is an Atom object and bonds is an indicator list.
                 Reference atom position and bond connections.
        '''

        for r, p in zip(ref, pred):
            
            pred_xyz = p[0].array(xyz=True)
            pred_class = p[0].class_index
            pred_edges = np.array(p[1])

            ref_xyz = [ri[0].array(xyz=True) for ri in r if ri[0].class_index > 0]

            if len(ref_xyz) > 0:

                closest_ind = np.argmin(np.linalg.norm(ref_xyz - pred_xyz, axis=1))
                ref_xyz = ref_xyz[closest_ind]
                ref_class = r[closest_ind][0].class_index
                ref_edges = np.array(r[closest_ind][1])
            
                # Position error
                abs_err = np.abs(pred_xyz - ref_xyz)
                eucl_dist = np.sqrt((abs_err**2).sum())
                self._pos_errors.append(np.append([eucl_dist], abs_err, axis=0)[None,:]) 
                self._ref_pos.append(ref_xyz[None, :])
                self._pred_pos.append(pred_xyz[None, :])
                
                # Edge classification confusion matrix
                if len(ref_edges) > 0:
                    pred_edge_classes = np.zeros_like(pred_edges, dtype=np.int32)
                    pred_edge_classes[pred_edges >= self.edge_conf] = 1
                    np.add.at(self.conf_mat_edge, (ref_edges, pred_edge_classes), 1)
                
            else:
                ref_class = 0
            
            # Node classification confusion matrix
            self.conf_mat_node[ref_class, pred_class] += 1

    def add_batch_grid(self, pred, ref):
        '''
        Gather stats from one batch of predictions and references of grid model.
        Arguments:
            pred: list of tuples (atom, bond, pos_dist), where atom is an Atom object, bond is a list,
                and pos_dist is np.ndarray of shape (x, y, z).
            ref: list of tuples (atom, bond, terminate), where atom is an Atom object, bond is a list, and
                terminate is an int.
        '''

        for r, p in zip(ref, pred):

            terminate = r[2]

            if not terminate:
            
                # Node classification confusion matrix
                pred_class = p[0].class_index
                ref_class = r[0].class_index
                self.conf_mat_node[ref_class, pred_class] += 1

                # Edge classification confusion matrix
                pred_edges = np.array(p[1])
                ref_edges = np.array(r[1])
                if len(ref_edges) > 0:
                    pred_edge_classes = np.zeros_like(pred_edges, dtype=np.int32)
                    pred_edge_classes[pred_edges >= self.edge_conf] = 1
                    np.add.at(self.conf_mat_edge, (ref_edges, pred_edge_classes), 1)   
        
    def plot(self, outdir='./', verbose=1):
        '''
        Plot confusion matrices and position error information for samples in gathered batches.
        Arguments:
            outdir: str. Directory where images are saved.
            verbose: 0 or 1. Whether to print information.
        '''
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Node classification
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        tick_labels = [f'Class {i}' for i in range(self.n_classes)]
        vis.plot_confusion_matrix(ax, self.conf_mat_node, tick_labels)
        ax.set_title('Node confusion matrix')
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'conf_mat_node.png'))
        if verbose > 0: print(f'Node confusion matrix saved to {savepath}')

        # Edge classification
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        tick_labels = ['No edge', 'Edge']
        vis.plot_confusion_matrix(ax, self.conf_mat_edge, tick_labels)
        ax.set_title('Edge confusion matrix')
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'conf_mat_edge.png'))
        if verbose > 0: print(f'Edge confusion matrix saved to {savepath}')
        
        if len(self._pos_errors) > 0:

            # Histograms of position errors
            fig, axes = plt.subplots(2, 2, figsize=(7, 8))
            fig.suptitle('Position Error')
            pos_errors = self.pos_errors
            for i, (pos, label, ax) in enumerate(zip(self.pos_errors.T, 'rxyz', axes.flatten())):
                ax.hist(pos, bins=20, edgecolor='black', density=True)
                ax.set_xlabel(f'|{label}| ($\AA$)')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(savepath := os.path.join(outdir, 'pos_error.png'))
            if verbose > 0: print(f'Position error histogram saved to {savepath}')
            
            # Plots of position errors as a function of reference atom depth
            mean_errors, bins = self.pos_depth_error()
            bins = bins[1:] + (bins[:-1] - bins[1:])/2
            labels = [r'$r$', r'$x$', r'$y$', r'$z$']
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            for i, (label, ax) in enumerate(zip(labels, axes.flatten())):
                ax.plot(bins, mean_errors[:,i])
                ax.set_xlabel('Ref atom depth')
                ax.set_ylabel(f'Mean abs error in {label} ($\AA$)')
            fig.suptitle('Position errors at different depths')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(savepath := os.path.join(outdir, 'pos_errors_depth.png'))
            if verbose > 0: print(f'Position errors as functions of depth saved to {savepath}')
            
            # Scatter plots of reference versus predicted position in x, y, and z
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))
            labels = [r'$x$', r'$y$', r'$z$']
            for r, p, label, ax in zip(self.ref_pos.T, self.pred_pos.T, labels, axes.flatten()):
                start, end = r.min(), r.max()
                ax.plot([start, end], [start, end])
                ax.scatter(r, p, c='r', s=1)
                ax.set_xlabel(f'Reference {label} ($\AA$)')
                ax.set_ylabel(f'Predicted {label} ($\AA$)')
            fig.suptitle('Reference vs. predicted positions')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(savepath := os.path.join(outdir, 'pos_scatter.png'))
            if verbose > 0: print(f'Scatter plots of ref vs pred position saved to {savepath}')

    def report(self, outdir='./', percentiles=[90, 95, 99], verbose=1):
        '''
        Save statistics of samples in gathered batches into text files.
        Arguments:
            outdir: str. Directory where files are saved.
            percentiles: list of ints. Percentile values reported for position errors.
            verbose: int 0 or 1. Whether to print output information.
        Saved files:
            pos_predictions.csv: Complete list of predictions, references and absolute errors in r, x, y, and z
            stats_pos.csv: Statistical values for position errors.
            stats_node.csv: Precision and recall for node classification
            stats_edge.csv: Precision and recall for edge classification
            stats_accuracy.csv: Overall accuracies of node and edge classification
            conf_mat_node.csv: Confusion matrix for node classification
            conf_mat_edge.csv: Confusion matrix for edge classification
        '''

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if len(self._pos_errors) > 0:
        
            # Position predictions list
            np.savetxt(
                savepath := os.path.join(outdir, 'pos_predictions.csv'),
                np.concatenate([self.ref_pos, self.pred_pos, self.pos_errors], axis=1),
                delimiter=',',
                comments=''
            )
            if verbose > 0: print(f'Position prediction list saved to {savepath}')
            
            # Position stats
            with open(os.path.join(outdir, 'stats_pos.csv'), 'w') as f:
                f.write('Position errors,|r|,|x|,|y|,|z|\n')
                f.write('1,angstrom,angstrom,angstrom,angstrom\n')
                for label, stat in zip(['Mean', 'Median', 'Std'], self.pos_stats):
                    f.write(f'{label}, {",".join([f"{s:.4f}" for s in stat])}\n')
                for p, perc in zip(percentiles, self.pos_percentiles(percentiles)):
                    f.write(f'{p}th percentile, {",".join([f"{p:.4f}" for p in perc])}\n')
            if verbose > 0: print(f'Position prediction stats saved to {f.name}')
            
        # Node precision and recall
        with open(os.path.join(outdir, 'stats_node.csv'), 'w') as f:
            f.write('Ref class,Precision,Recall\n')
            for i, (prec, rec) in enumerate(zip(self.node_precision, self.node_recall)):
                f.write(f'{i},{prec:.4f},{rec:.4f}')
                if i < self.n_classes-1:
                    f.write('\n')
        if verbose > 0: print(f'Node prediction stats saved to {f.name}')
                
        # Edge precision and recall
        with open(os.path.join(outdir, 'stats_edge.csv'), 'w') as f:
            f.write('Ref class,Precision,Recall\n')
            for label, prec, rec in zip(['No edge', 'Edge'], self.edge_precision, self.edge_recall):
                f.write(f'{label},{prec:.4f},{rec:.4f}')
                if label == 'No edge':
                    f.write('\n')
        if verbose > 0: print(f'Edge prediction stats saved to {f.name}')
                
        # Accuracies
        with open(os.path.join(outdir, 'stats_accuracy.csv'), 'w') as f:
            f.write('Node accuracy,Edge accuracy\n')
            f.write(f'{self.node_accuracy:.4f},{self.edge_accuracy:.4f}')
                
        if verbose > 0: print(f'Node and edge prediction accuracies saved to {f.name}')
        
        # Node confusion matrix
        np.savetxt(
            savepath := os.path.join(outdir, 'conf_mat_node.csv'),
            self.conf_mat_node,
            delimiter=','
        )
        if verbose > 0: print(f'Node confusion matrix data saved to {savepath}')
        
        # Edge confusion matrix
        np.savetxt(
            savepath := os.path.join(outdir, 'conf_mat_edge.csv'),
            self.conf_mat_edge,
            delimiter=','
        )
        if verbose > 0: print(f'Edge confusion matrix data saved to {savepath}')

class GraphSeqStats():
    '''
    Gather statistics on graph sequence predictions.
    '''
    def __init__(self, n_classes, dist_threshold=0.35, bin_size=4):
        self.n_classes = n_classes
        self.bin_size = bin_size
        self.n_bins = 1
        self._graph_sizes = [[]]
        self._node_count_diffs = [[]]
        self._bond_count_diffs = [[]]
        self._hausdorff_distances = [[]]
        self._matching_distances = [[]]
        self._conf_mat_node = [np.zeros((n_classes, n_classes), dtype=np.int32)]
        self._conf_mat_edge = [np.zeros((2, 2), dtype=np.int32)]
        self._missing_nodes = [[]]
        self._extra_nodes = [[]]
        self.dist_threshold = dist_threshold

    @property
    def largest_graph(self):
        return max(self.graph_sizes())

    @property
    def total_nodes(self):
        return sum(self.graph_sizes())

    @property
    def total_samples(self):
        return len(self.graph_sizes())

    def conf_mat_node(self, size_bin=-1):
        return np.sum(self._conf_mat_node, axis=0) if size_bin < 0 else self._conf_mat_node[size_bin]
    
    def conf_mat_edge(self, size_bin=-1):
        return np.sum(self._conf_mat_edge, axis=0) if size_bin < 0 else self._conf_mat_edge[size_bin]

    def edge_precision(self, size_bin=-1):
        conf_mat = self.conf_mat_edge(size_bin)
        return np.diag(conf_mat) / conf_mat.sum(axis=0)

    def edge_recall(self, size_bin=-1):
        conf_mat = self.conf_mat_edge(size_bin)
        return np.diag(conf_mat) / conf_mat.sum(axis=1)
    
    def node_precision(self, size_bin=-1):
        conf_mat = self.conf_mat_node(size_bin)
        return np.diag(conf_mat) / conf_mat.sum(axis=0)
    
    def node_recall(self, size_bin=-1):
        conf_mat = self.conf_mat_node(size_bin)
        return np.diag(conf_mat) / conf_mat.sum(axis=1)

    def _get_array(self, arrays, size_bin):
        arrays = [np.array(a) for a in arrays]
        return np.concatenate(arrays, axis=0) if size_bin < 0 else arrays[size_bin]

    def graph_sizes(self, size_bin=-1):
        return self._get_array(self._graph_sizes, size_bin)

    def node_count_diffs(self, size_bin=-1):
        return self._get_array(self._node_count_diffs, size_bin)

    def bond_count_diffs(self, size_bin=-1):
        return self._get_array(self._bond_count_diffs, size_bin)

    def hausdorff_distances(self, size_bin=-1):
        return self._get_array(self._hausdorff_distances, size_bin)

    def matching_distances(self, size_bin=-1):
        return self._get_array(self._matching_distances, size_bin)

    def missing_nodes(self, size_bin=-1):
        return self._get_array(self._missing_nodes, size_bin)

    def extra_nodes(self, size_bin=-1):
        return self._get_array(self._extra_nodes, size_bin)

    def _check_bins(self, size_bin):
        for _ in range(self.n_bins, size_bin+1):
            self._graph_sizes.append([])
            self._node_count_diffs.append([])
            self._bond_count_diffs.append([])
            self._hausdorff_distances.append([])
            self._matching_distances.append([])
            self._conf_mat_node.append(np.zeros((self.n_classes, self.n_classes), dtype=np.int32))
            self._conf_mat_edge.append(np.zeros((2, 2), dtype=np.int32))
            self._missing_nodes.append([])
            self._extra_nodes.append([])
            self.n_bins += 1

    def add_batch(self, pred, ref):
        '''
        Gather stats from one batch of predictions and references.
        Arguments:
            pred: list of MoleculeGraph. Predicted molecule graphs.
            ref: list of MoleculeGraph. Reference molecule graphs.
        '''

        assert len(pred) == len(ref), 'Different number of predictions and references.'

        for p, r in zip(pred, ref):

            graph_size = len(r)
            size_bin = (graph_size - 1) // self.bin_size
            self._check_bins(size_bin)
            self._graph_sizes[size_bin].append(graph_size)
            
            # Node and bond count diffs
            self._node_count_diffs[size_bin].append(len(p) - len(r))
            self._bond_count_diffs[size_bin].append(len(p.bonds) - len(r.bonds))
            
            if len(p.atoms) > 0:

                pos1 = p.array(xyz=True)
                pos2 = r.array(xyz=True)

                # Hausdorff distance
                d1 = directed_hausdorff(pos1, pos2)
                d2 = directed_hausdorff(pos2, pos1)
                self._hausdorff_distances[size_bin].append(max(d1[0], d2[0]))

                # Match closest positions in prediction and reference
                dist_mat = cdist(pos2, pos1, metric='euclidean')
                mapping = []
                missing_nodes = []
                for i, dists in enumerate(dist_mat):
                    matches = np.where(dists < self.dist_threshold)[0]
                    if len(matches) == 0:
                        missing_nodes.append(i)
                    elif len(matches) == 1:
                        mapping.append(matches[0])
                        self._matching_distances[size_bin].append(dists[matches[0]])
                    else:
                        mapping.append(matches[np.argmin(dists[matches])])
                        self._matching_distances[size_bin].append(dists[mapping[-1]])
                p_extra_nodes = list(set(range(len(p))) - set(mapping))
                n_matches = len(mapping)
                mapping += p_extra_nodes
                self._missing_nodes[size_bin].append(len(missing_nodes))
                self._extra_nodes[size_bin].append(len(p_extra_nodes))
                
                # Prune graphs to match nodes one-to-one
                r_pruned = r.remove_atoms(missing_nodes)[0]
                p_pruned = p.permute(mapping).remove_atoms([n_matches + i for i in range(len(p_extra_nodes))])[0]
                assert len(r_pruned) == len(p_pruned), f'{len(r_pruned)}, {len(p_pruned)}'

                # Node confusion matrix
                for ri, pi in zip(r_pruned.atoms, p_pruned.atoms):
                    self._conf_mat_node[size_bin][ri.class_index, pi.class_index] += 1

                # Edge confusion matrix
                Ar = r_pruned.adjacency_matrix()
                Ap = p_pruned.adjacency_matrix()
                Ar = Ar[np.triu_indices(len(Ar), k=1)].flatten()
                Ap = Ap[np.triu_indices(len(Ap), k=1)].flatten()
                np.add.at(self._conf_mat_edge[size_bin], (Ar, Ap), 1)          

    def plot(self, outdir='./', verbose=1):
        '''
        Plot histograms of graph sizes, node/bond count differences, hausdorff distances, and maching distances,
        and confusion matrices for node and edge classification.
        Arguments:
            outdir: str. Directory where images are saved.
            verbose: 0 or 1. Whether to print information.
        '''

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        def count_histogram(counts):
            fig = plt.figure(figsize=(8, 8))
            bin_min = np.min(counts) - 0.5
            bin_max = np.max(counts) + 1.5
            bins = np.arange(bin_min, bin_max, 1)
            plt.hist(counts, bins=bins, edgecolor='black', density=True)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Histogram of graph sizes
        count_histogram(self.graph_sizes())
        plt.title('Reference graph size')
        plt.savefig(savepath := os.path.join(outdir, 'graph_size.png'))
        if verbose > 0: print(f'Graph size histogram saved to {savepath}')
        plt.close()

        # Histogram of node count diffs
        count_histogram(self.node_count_diffs())
        plt.title('Node count difference')
        plt.savefig(savepath := os.path.join(outdir, 'node_diff.png'))
        if verbose > 0: print(f'Node count difference histogram saved to {savepath}')
        plt.close()

        # Histogram of bond count diffs
        count_histogram(self.bond_count_diffs())
        plt.title('Bond count difference')
        plt.savefig(savepath := os.path.join(outdir, 'bond_diff.png'))
        if verbose > 0: print(f'Bond count difference histogram saved to {savepath}')
        plt.close()

        # Node classification
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        tick_labels = [f'Class {i}' for i in range(self.n_classes)]
        vis.plot_confusion_matrix(ax, self.conf_mat_node(), tick_labels)
        ax.set_title('Node confusion matrix')
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'conf_mat_node.png'))
        if verbose > 0: print(f'Node confusion matrix saved to {savepath}')
        plt.close()

        # Edge classification
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        tick_labels = ['No edge', 'Edge']
        vis.plot_confusion_matrix(ax, self.conf_mat_edge(), tick_labels)
        ax.set_title('Edge confusion matrix')
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'conf_mat_edge.png'))
        if verbose > 0: print(f'Edge confusion matrix saved to {savepath}')
        plt.close()

        # Binned node precision
        fig = plt.figure(figsize=(6, 5))
        bins = [(i+1)*self.bin_size for i in range(self.n_bins)]
        for c in range(self.n_classes):
            prec = [self.node_precision(b)[c] for b in range(self.n_bins)]
            plt.plot(bins, prec)
        plt.legend([str(c) for c in range(self.n_classes)])
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'binned_node_precision.png'))
        if verbose > 0: print(f'Plot of node precision as a function of graph size saved to {savepath}')
        plt.close()

        # Binned node recall
        fig = plt.figure(figsize=(6, 5))
        bins = [(i+1)*self.bin_size for i in range(self.n_bins)]
        for c in range(self.n_classes):
            prec = [self.node_recall(b)[c] for b in range(self.n_bins)]
            plt.plot(bins, prec)
        plt.legend([str(c) for i in range(self.n_classes)])
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'binned_node_recall.png'))
        if verbose > 0: print(f'Plot of node recall as a function of graph size saved to {savepath}')
        plt.close()

        # Binned edge precision
        fig = plt.figure(figsize=(6, 5))
        bins = [(i+1)*self.bin_size for i in range(self.n_bins)]
        for c in range(2):
            prec = [self.edge_precision(b)[c] for b in range(self.n_bins)]
            plt.plot(bins, prec)
        plt.legend(['No edge', 'Edge'])
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'binned_edge_precision.png'))
        if verbose > 0: print(f'Plot of edge precision as a function of graph size saved to {savepath}')
        plt.close()

        # Binned edge recall
        fig = plt.figure(figsize=(6, 5))
        bins = [(i+1)*self.bin_size for i in range(self.n_bins)]
        for c in range(2):
            prec = [self.edge_recall(b)[c] for b in range(self.n_bins)]
            plt.plot(bins, prec)
        plt.legend(['No edge', 'Edge'])
        plt.tight_layout()
        plt.savefig(savepath := os.path.join(outdir, 'binned_edge_recall.png'))
        if verbose > 0: print(f'Plot of edge recall as a function of graph size saved to {savepath}')
        plt.close()

        if len(self.hausdorff_distances()) > 0:

            # Histogram of Hausdorff distances
            fig = plt.figure(figsize=(8, 8))
            plt.hist(self.hausdorff_distances(), bins=20, edgecolor='black', density=True)
            plt.title('Hausdorff distances')
            plt.xlabel(f'Distance ($\AA$)')
            plt.savefig(savepath := os.path.join(outdir, 'hausdorff.png'))
            if verbose > 0: print(f'Hausdorff distance histogram saved to {savepath}')
            plt.close()

            # Histograms of matching distances
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle('Matching distance')
            plt.hist(self.matching_distances(), bins=20, edgecolor='black', density=True)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(savepath := os.path.join(outdir, 'matching_distances.png'))
            if verbose > 0: print(f'Histogram of matching distances saved to {savepath}')
            plt.close()

    def report(self, outdir='./', verbose=1):
        '''
        Save to file mean absolute node/bond count diffs, mean hausdorff, mean matching distance,
        missing/extra atoms, total samples/nodes, average/largest graph size, and
        node/edge precision/recall.
        Arguments:
            outdir: str. Directory where files are saved.
            verbose: 0 or 1. Whether to print information.
        '''

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(os.path.join(outdir, 'seq_stats.csv'), 'w') as f:
            f.write(f'Mean absolute node diff, {np.abs(self.node_count_diffs()).mean()}\n')
            f.write(f'Mean absolute bond diff, {np.abs(self.bond_count_diffs()).mean()}\n')
            f.write(f'Mean Hausdorff distance, {np.mean(self.hausdorff_distances())}\n')
            f.write(f'Mean matching distance, {np.mean(self.matching_distances())}\n')
            f.write(f'Average missing atoms, {np.mean(self.missing_nodes())}\n')
            f.write(f'Average extra atoms, {np.mean(self.extra_nodes())}\n')
            f.write(f'Total samples, {self.total_samples}\n')
            f.write(f'Total nodes, {self.total_nodes}\n')
            f.write(f'Average graph size, {self.total_nodes / self.total_samples}\n')
            f.write(f'Largest graph size, {self.largest_graph}\n')
        if verbose > 0: print(f'Sequence stats saved to {f.name}')

        with open(os.path.join(outdir, 'graph_sizes.csv'), 'w') as f:
            f.write(','.join([str(s) for s in self.graph_sizes()]))

        # Node precision and recall
        with open(os.path.join(outdir, 'stats_node.csv'), 'w') as f:
            f.write('Ref class,Precision,Recall\n')
            for i, (prec, rec) in enumerate(zip(self.node_precision(), self.node_recall())):
                f.write(f'{i},{prec:.4f},{rec:.4f}')
                if i < self.n_classes-1:
                    f.write('\n')
        if verbose > 0: print(f'Sequence node prediction stats saved to {f.name}')

        # Edge precision and recall
        with open(os.path.join(outdir, 'stats_edge.csv'), 'w') as f:
            f.write('Ref class,Precision,Recall\n')
            for label, prec, rec in zip(['No edge', 'Edge'], self.edge_precision(), self.edge_recall()):
                f.write(f'{label},{prec:.4f},{rec:.4f}')
                if label == 'No edge':
                    f.write('\n')
        if verbose > 0: print(f'Sequence edge prediction stats saved to {f.name}')

        # Node confusion matrix
        np.savetxt(
            savepath := os.path.join(outdir, 'conf_mat_node.csv'),
            self.conf_mat_node(),
            delimiter=','
        )
        if verbose > 0: print(f'Node confusion matrix data saved to {savepath}')
        
        # Edge confusion matrix
        np.savetxt(
            savepath := os.path.join(outdir, 'conf_mat_edge.csv'),
            self.conf_mat_edge(),
            delimiter=','
        )
        if verbose > 0: print(f'Edge confusion matrix data saved to {savepath}')

        # Binned stats
        with open(os.path.join(outdir, 'binned_seq_stats.csv'), 'w') as f:
            f.write(',' + ','.join([str((i+1)*self.bin_size) for i in range(self.n_bins)]) + '\n')
            f.write('Number of samples,' + ','.join([str(len(self.graph_sizes(b))) for b in range(self.n_bins)]) + '\n')
            f.write('Mean absolute node diff,' +
                ','.join([str(np.abs(self.node_count_diffs(b)).mean()) for b in range(self.n_bins)]) + '\n')
            f.write('Mean absolute bond diff,' +
                ','.join([str(np.abs(self.bond_count_diffs(b)).mean()) for b in range(self.n_bins)]) + '\n')
            f.write('Mean Hausdorff distance,' +
                ','.join([str(self.hausdorff_distances(b).mean()) for b in range(self.n_bins)]) + '\n')
            f.write('Mean matching distance,' +
                ','.join([str(self.matching_distances(b).mean()) for b in range(self.n_bins)]) + '\n')
            f.write('Missing atoms (mean),' +
                ','.join([str(self.missing_nodes(b).mean()) for b in range(self.n_bins)]) + '\n')
            f.write('Missing atoms (std),' +
                ','.join([str(self.missing_nodes(b).std()) for b in range(self.n_bins)]) + '\n')
            f.write('Extra atoms (mean),' +
                ','.join([str(self.extra_nodes(b).mean()) for b in range(self.n_bins)]) + '\n')
            f.write('Extra atoms (std),' +
                ','.join([str(self.extra_nodes(b).std()) for b in range(self.n_bins)]))

        # Binned node precision and recall
        with open(os.path.join(outdir, 'binned_node_recall.csv'), 'w') as f1, open(os.path.join(outdir, 'binned_node_precision.csv'), 'w') as f2:
            f1.write('Ref class,' + ','.join([str((i+1)*self.bin_size) for i in range(self.n_bins)]) + '\n')
            f2.write('Ref class,' + ','.join([str((i+1)*self.bin_size) for i in range(self.n_bins)]) + '\n')
            for c in range(self.n_classes):
                f1.write(f'{c},')
                f2.write(f'{c},')
                for b in range(self.n_bins):
                    f1.write(f'{self.node_recall(b)[c]}')
                    f2.write(f'{self.node_precision(b)[c]}')
                    if b < self.n_bins-1:
                        f1.write(',')
                        f2.write(',')
                    elif c < self.n_classes-1:
                        f1.write('\n')
                        f2.write('\n')
        
        # Binned edge precision and recall
        with open(os.path.join(outdir, 'binned_edge_recall.csv'), 'w') as f1, open(os.path.join(outdir, 'binned_edge_precision.csv'), 'w') as f2:
            f1.write('Ref class,' + ','.join([str((i+1)*self.bin_size) for i in range(self.n_bins)]) + '\n')
            f2.write('Ref class,' + ','.join([str((i+1)*self.bin_size) for i in range(self.n_bins)]) + '\n')
            for c, label in enumerate(['No edge', 'Edge']):
                f1.write(f'{label},')
                f2.write(f'{label},')
                for b in range(self.n_bins):
                    f1.write(f'{self.edge_recall(b)[c]}')
                    f2.write(f'{self.edge_precision(b)[c]}')
                    if b < self.n_bins-1:
                        f1.write(',')
                        f2.write(',')
                    elif c == 0:
                        f1.write('\n')
                        f2.write('\n')

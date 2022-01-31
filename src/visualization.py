
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, gridspec

from utils import _calc_plot_dim, elements

CLASS_COLORS = 'rkbgcmy'

def _get_mol_bounding_box(mol, pred, ref, margin=0.5):
    mol = mol[:,:3]
    mol = np.append(mol, pred[0][None,:3], axis=0)
    mol = np.append(mol, ref[0][None,:3], axis=0)
    lims_min = np.min(mol, axis=0)
    lims_max = np.max(mol, axis=0)
    max_range = (lims_max-lims_min).max()
    center = (lims_min + lims_max) / 2
    radius = max_range / 2
    lims = np.stack([center - radius - margin, center + radius + margin]).T
    return lims

def _get_mol_bounding_box_multiple(mol_xyz, pred_xyz, ref_xyz, margin=0.5):
    xyz = np.concatenate([mol_xyz, pred_xyz, ref_xyz], axis=0)
    lims_min = np.min(xyz, axis=0)
    lims_max = np.max(xyz, axis=0)
    max_range = (lims_max-lims_min).max()
    center = (lims_min + lims_max) / 2
    radius = max_range / 2
    lims = np.stack([center - radius - margin, center + radius + margin]).T
    return lims

def plot_input(X, constant_range=False, cmap='afmhot'):
    '''
    Plot single stack of AFM images.
    Arguments:
        X: np.ndarray of shape (x, y, z). AFM image to plot.
        constant_range: Boolean. Whether the different slices should use the same value range or not.
        cmap: str or matplotlib colormap. Colormap to use for plotting.
    Returns: matplotlib.pyplot.figure. Figure on which the image was plotted.
    '''
    rows, cols = _calc_plot_dim(X.shape[-1])
    fig = plt.figure(figsize=(3.2*cols,2.5*rows))
    vmax = X.max()
    vmin = X.min()
    for k in range(X.shape[-1]):
        fig.add_subplot(rows,cols,k+1)
        if constant_range:
            plt.imshow(X[:,:,k].T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        else:
            plt.imshow(X[:,:,k].T, cmap=cmap, origin="lower")
        plt.colorbar()
    plt.tight_layout()
    return fig
        
def make_input_plots(Xs, outdir='./predictions/', start_ind=0, constant_range=False, cmap='afmhot', verbose=1):
    '''
    Plot multiple AFM images to files 0_input.png, 1_input.png, ... etc.
    Arguments:
        Xs: list of np.ndarray of shape (batch, x, y, z). Input AFM images to plot.
        outdir: str. Directory where images are saved.
        start_ind: int. Save index increments by one for each image. The first index is start_ind.
        constant_range: Boolean. Whether the different slices should use the same value range or not.
        cmap: str or matplotlib colormap. Colormap to use for plotting.
        verbose: int 0 or 1. Whether to print output information.
    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    img_ind = start_ind
    for i in range(Xs[0].shape[0]):
        
        for j in range(len(Xs)):
            
            plot_input(Xs[j][i], constant_range, cmap=cmap)
            
            save_name = f'{img_ind}_input'
            if len(Xs) > 1:
                save_name += str(j+1)
            save_name = os.path.join(outdir, save_name)
            save_name += '.png'
            plt.savefig(save_name)
            plt.close()

            if verbose > 0: print(f'Input image saved to {save_name}')

        img_ind += 1

def plot_confusion_matrix(ax, conf_mat, tick_labels=None):
    '''
    Plot confusion matrix on matplotlib axes.
    Arguments:
        ax: matplotlib.axes.Axes. Axes object on which the confusion matrix is plotted.
        conf_mat: np.ndarray of shape (num_classes, num_classes). Confusion matrix counts.
        tick_labels: list of str. Labels for classes.
    '''
    if tick_labels:
        assert len(conf_mat) == len(tick_labels)
    else:
        tick_labels = [str(i) for i in range(len(conf_mat))]
        
    conf_mat_norm = np.zeros_like(conf_mat, dtype=np.float64)
    for i, r in enumerate(conf_mat):
        conf_mat_norm[i] = r / np.sum(r)
    
    im = ax.imshow(conf_mat_norm, cmap=cm.Blues)
    plt.colorbar(im)
    ax.set_xticks(np.arange(conf_mat.shape[0]))
    ax.set_yticks(np.arange(conf_mat.shape[1]))
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels, rotation='vertical', va='center')
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            color = 'white' if conf_mat_norm[i,j] > 0.5 else 'black'
            label = '{:.3f}'.format(conf_mat_norm[i,j])+'\n('+'{:d}'.format(conf_mat[i,j])+')'
            ax.text(j, i, label, ha='center', va='center', color=color)


def plot_graph_prediction_grid(pred, ref, mol, box_borders):
    '''
    Plot 3D view and 2D top view of graph node prediction and reference.

    Arguments:
        pred: tuple (atom, bond, pred_dist), where atom is an Atom object, bond is a list,
            and pred_dist is np.ndarray of shape (x, y, z).
        ref: tuple (atom, bond, terminate, ref_dist), where atom is an Atom object, bond is a list, and
            terminate is an int, and ref_dist is np.ndarray of shape (x, y, z). ref_dist is optional.
        mol: MoleculeGraph. Input molecules.
        box_borders: tuple ((x_start, y_start, z_start),(x_end, y_end, z_end)). Position of plotting region.
    
    Returns: matplotlib.pyplot.Figure.
    '''

    fig = plt.figure(figsize=(10,10))

    ref_dist = ref[3].mean(axis=-1)
    pred_dist = pred[2].mean(axis=-1)
    vmin = min(ref_dist.min(), pred_dist.min())
    vmax = max(ref_dist.max(), pred_dist.max())
    mol_pos = mol.array(xyz=True) if len(mol) > 0 else np.empty((0,3))

    # Suppress atoms from a terminated graph
    ref, pred = (ref, pred) if ref[2] == 0 else (None, None)

    def plot(i, atom, bonds, label, pd):
        
        ax_grid = fig.add_subplot(221+i)
        ax_graph = fig.add_subplot(223+i)

        atom_pos = atom.array(xyz=True) if atom else []
        
        z_min, z_max = box_borders[0][2], box_borders[1][2]
        s = 80*(mol_pos[:,2] - z_min) / (z_max - z_min)
        if (s < 0).any():
            raise ValueError('Encountered atom z position(s) below box borders.')

        extent = [box_borders[0][0], box_borders[1][0], box_borders[0][1], box_borders[1][1]]
        ax_grid.imshow(pd.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
        
        ax_graph.scatter(mol_pos[:,0], mol_pos[:,1], c='gray', s=s)
        for b in mol.bonds:
            pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
            ax_graph.plot(pos[:,0], pos[:,1], 'gray')
        if len(atom_pos) > 0:
            s = 80*(atom_pos[2] - z_min) / (z_max - z_min)
            ax_graph.scatter(atom_pos[0], atom_pos[1], c=CLASS_COLORS[atom.class_index], s=s)
            for b in bonds:
                pos = np.vstack([mol_pos[b,:3], atom_pos])
                ax_graph.plot(pos[:,0], pos[:,1], c='r')
        
        ax_graph.set_xlim(box_borders[0][0], box_borders[1][0])
        ax_graph.set_ylim(box_borders[0][1], box_borders[1][1])
        ax_graph.set_title(f'{label}')

        return ax_grid, ax_graph

    # Prediction
    pred_atom, pred_bonds = (pred[0], pred[1]) if pred else (None, [])
    plot(0, pred_atom, pred_bonds, 'Prediction', pred_dist)

    # Reference
    ref_atom, ref_bonds = (ref[0], ref[1]) if ref else (None, [])
    ax_grid, _ = plot(1, ref_atom, ref_bonds, 'Reference', ref_dist)
        
    # Colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    pos = ax_grid.get_position()
    cax = fig.add_axes(rect=[0.9, pos.ymin, 0.03, pos.ymax - pos.ymin])
    m = cm.ScalarMappable()
    m.set_array([vmin, vmax])
    plt.colorbar(m, cax=cax)
        
    return fig

def visualize_graph_grid(molecules, pred, ref, bond_threshold=0.5, box_borders=((2,2,-1.5),(18,18,0)),
    outdir='./graphs/', start_ind=0, show=False, verbose=1):
    '''
    Plot grid model single-step predictions.

    Arguments: 
        molecules: list of MoleculeGraph. Input molecule graphs.
        pred: list of tuples (atom, bond, pred_dist), where atom is an Atom object, bond is a list,
            and pred_dist is np.ndarray of shape (x, y, z).
        ref: list of tuples (atom, bond, terminate, ref_dist), where atom is an Atom object, bond is a list,
            terminate is an int, and ref_dist is np.ndarray of shape (x, y, z).
        bond_threshold: float in [0,1]. Predicted bonds with confidence level above bond_threshold are plotted.
        box_borders: tuple ((x_start, y_start, z_start),(x_end, y_end, z_end)). Position of plotting region.
        outdir: str. Directory where images are saved.
        start_ind: int. Save index increments by one for each graph. The first index is start_ind.
        show: Boolean. whether to show an interactive window for each graph.
        verbose: int 0 or 1. Whether to print output information.
    '''
    
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Convert bond indicator lists into index lists
    pred = [(a, np.where(b >= np.array(bond_threshold))[0], p) for a, b, p in pred]
    ref = [(a, np.where(b)[0], t, d) for a, b, t, d in ref]

    counter = start_ind
    for mol, p, r in zip(molecules, pred, ref):
    
        plot_graph_prediction_grid(p, r, mol, box_borders)
        
        if outdir:
            plt.savefig(save_path:=os.path.join(outdir, f'{counter}_pred_graph.png'))
            if verbose > 0: print(f'Graph image saved to {save_path}.')
        if show:
            plt.show()
        plt.close()
        
        counter += 1

def plot_graph_sequence_grid(pred_graphs, ref_graphs, pred_sequence, box_borders=((2,2,-1.5),(18,18,0)),
    classes=None, class_colors=CLASS_COLORS, z_min=None, z_max=None, outdir='./graphs/', start_ind=0, verbose=1):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if classes:
        assert len(class_colors) >= len(classes), f'Not enough colors for classes'

    if z_min is None: z_min = box_borders[0][2]
    if z_max is None: z_max = box_borders[1][2]
    scatter_size = 160

    def get_marker_size(z, max_size):
        return max_size * (z - z_min) / (z_max - z_min)

    def plot_xy(ax, atom_pos, mol, atom, bonds, scatter_size):
        
        if atom_pos is not None:
            s = get_marker_size(atom_pos[:,2], scatter_size)
            ax.scatter(atom_pos[:,0], atom_pos[:,1], c='lightgray', s=s)

        if len(mol) > 0:

            mol_pos = mol.array(xyz=True)

            s = get_marker_size(mol_pos[:,2], scatter_size)
            if (s < 0).any():
                raise ValueError('Encountered atom z position(s) below box borders.')
            
            c = [class_colors[atom.class_index] for atom in mol.atoms]

            ax.scatter(mol_pos[:,0], mol_pos[:,1], c=c, s=s, edgecolors='k', zorder=2)
            for b in mol.bonds:
                pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
                ax.plot(pos[:,0], pos[:,1], 'k', linewidth=2, zorder=1)

        if atom is not None:
            atom_xyz = atom.array(xyz=True)
            c = class_colors[atom.class_index]
            s = get_marker_size(atom_xyz[2], scatter_size)
            ax.scatter(atom_xyz[0], atom_xyz[1], c=c, s=s, edgecolors='k', zorder=2)
            bonds = [i for i in range(len(bonds)) if bonds[i] > 0.5]
            for b in bonds:
                pos = np.vstack([mol_pos[b], atom_xyz])
                ax.plot(pos[:,0], pos[:,1], 'k', linewidth=2, zorder=1)
        
        ax.set_xlim(box_borders[0][0], box_borders[1][0])
        ax.set_ylim(box_borders[0][1], box_borders[1][1])
        ax.set_aspect('equal', 'box')

    def plot_xz(ax, mol, scatter_size):

        if len(mol) > 0:

            order = list(np.argsort(mol.array(xyz=True)[:, 1])[::-1])
            mol = mol.permute(order)
            mol_pos = mol.array(xyz=True)

            s = get_marker_size(mol_pos[:,2], scatter_size)
            if (s < 0).any():
                raise ValueError('Encountered atom z position(s) below box borders.')
            
            c = [class_colors[atom.class_index] for atom in mol.atoms]

            for b in mol.bonds:
                pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
                ax.plot(pos[:,0], pos[:,2], 'k', linewidth=2, zorder=1)
            ax.scatter(mol_pos[:,0], mol_pos[:,2], c=c, s=s, edgecolors='k', zorder=2)
        
        ax.set_xlim(box_borders[0][0], box_borders[1][0])
        ax.set_ylim(box_borders[0][2], box_borders[1][2])
        ax.set_aspect('equal', 'box')

    ind = start_ind
    for p, r, s in zip(pred_graphs, ref_graphs, pred_sequence):

        # Setup plotting grid
        seq_len = len(s) + 1
        x_seq = min(6, seq_len)
        y_seq = int(seq_len / 6.1) + 1
        fig_seq = plt.figure(figsize=(2.5*x_seq, 2.8*y_seq))
        grid_seq = gridspec.GridSpec(y_seq, x_seq)

        # Plot prediction sequence
        atom_pos = p.array(xyz=True) if len(p) > 0 else None
        for i in range(len(s) + 1):
            if i == 0:
                mol, atom, bonds = [], None, None
            else:
                mol, atom, bonds = s[i-1]
            x_grid = i % x_seq
            y_grid = i // x_seq
            ax = fig_seq.add_subplot(grid_seq[y_grid, x_grid])
            plot_xy(ax, atom_pos, mol, atom, bonds, 100)
            ax.set_title(f'{i}')
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig_seq.tight_layout()

        plt.savefig(save_path:=os.path.join(outdir, f'{ind}_pred_sequence.png'))
        if verbose > 0: print(f'Graph prediction sequence image saved to {save_path}')
        plt.close()

        # Plot final graph
        if classes:
            x_extra = 0.35 * max([len(c) for c in classes])
            fig_final = plt.figure(figsize=(10+x_extra, 6.5))
            fig_grid = gridspec.GridSpec(1, 2, width_ratios=(10, x_extra), wspace=1/(10+x_extra))
        else:
            fig_final = plt.figure(figsize=(10, 6.5))
            fig_grid = gridspec.GridSpec(1, 1)
        grid_final = fig_grid[0, 0].subgridspec(2, 2, height_ratios=(5, 1.5), hspace=0.1, wspace=0.2)

        # Prediction
        ax_xy_pred = fig_final.add_subplot(grid_final[0, 0])
        ax_xz_pred = fig_final.add_subplot(grid_final[1, 0])
        plot_xy(ax_xy_pred, None, p, None, None, scatter_size)
        plot_xz(ax_xz_pred, p, scatter_size)
        ax_xy_pred.set_xlabel('x (Å)', fontsize=12)
        ax_xy_pred.set_ylabel('y (Å)', fontsize=12)
        ax_xz_pred.set_xlabel('x (Å)', fontsize=12)
        ax_xz_pred.set_ylabel('z (Å)', fontsize=12)
        ax_xy_pred.set_title('Prediction', fontsize=20)

        # Reference
        ax_xy_ref = fig_final.add_subplot(grid_final[0, 1])
        ax_xz_ref = fig_final.add_subplot(grid_final[1, 1])
        plot_xy(ax_xy_ref, None, r, None, None, scatter_size)
        plot_xz(ax_xz_ref, r, scatter_size)
        ax_xy_ref.set_xlabel('x (Å)', fontsize=12)
        ax_xy_ref.set_ylabel('y (Å)', fontsize=12)
        ax_xz_ref.set_xlabel('x (Å)', fontsize=12)
        ax_xz_ref.set_ylabel('z (Å)', fontsize=12)
        ax_xy_ref.set_title('Reference', fontsize=20)

        if classes:

            # Plot legend
            ax_legend = fig_final.add_subplot(fig_grid[0, 1])

            # Class colors
            dy = 0.08
            dx = 0.35 / x_extra
            y_start = 0.5 + dy * (len(classes) + 3) / 2
            for i, c in enumerate(classes):
                ax_legend.scatter(dx, y_start-dy*i, s=scatter_size, c=class_colors[i], edgecolors='k')
                ax_legend.text(2*dx, y_start-dy*i, ', '.join([elements[e-1] for e in c]), fontsize=16,
                    ha='left', va='center_baseline')

            # Marker sizes
            y_start2 = y_start - (len(classes) + 1) * dy
            marker_zs = np.array([z_max, (z_min + z_max + 0.2) / 2, z_min + 0.2])
            ss = get_marker_size(marker_zs, scatter_size)
            for i, (s, z) in enumerate(zip(ss, marker_zs)):
                ax_legend.scatter(dx, y_start2-dy*i, s=s, c='w', edgecolors='k')
                ax_legend.text(2*dx, y_start2-dy*i, f'z = {z}Å', fontsize=16,
                    ha='left', va='center_baseline')

            ax_legend.set_xlim(0, 1)
            ax_legend.set_ylim(0, 1)
            ax_legend.axis('off')

        plt.savefig(save_path:=os.path.join(outdir, f'{ind}_pred_final.png'))
        if verbose > 0: print(f'Final graph prediction image saved to {save_path}')
        plt.close()

        ind += 1

def plot_distribution_grid(pred_dist, ref_dist, box_borders=((2,2,-1.5),(18,18,0)),
    outdir='./graphs/', start_ind=0, verbose=1):

    assert pred_dist.shape == ref_dist.shape

    if not os.path.exists(outdir):
        os.makedirs(outdir)    

    fontsize = 24

    z_start = box_borders[0][2]
    z_res = (box_borders[1][2] - box_borders[0][2]) / pred_dist.shape[-1]
    extent = [box_borders[0][0], box_borders[1][0], box_borders[0][1], box_borders[1][1]]
    
    ind = start_ind
    for p, r in zip(pred_dist, ref_dist):

        # Plot grid in 2D
        p_mean, r_mean = p.mean(axis=-1), r.mean(axis=-1)
        vmin = min(r_mean.min(), p_mean.min())
        vmax = max(r_mean.max(), p_mean.max())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(p_mean.T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax2.imshow(r_mean.T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax1.set_title('Prediction')
        ax2.set_title('Reference')

        # Colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        pos = ax2.get_position()
        cax = fig.add_axes(rect=[0.9, pos.ymin, 0.03, pos.ymax - pos.ymin])
        m = cm.ScalarMappable()
        m.set_array([vmin, vmax])
        plt.colorbar(m, cax=cax)

        plt.savefig(save_path:=os.path.join(outdir, f'{ind}_pred_dist2D.png'))
        if verbose > 0: print(f'Position distribution 2D prediction image saved to {save_path}')
        plt.close()

        # Plot each z-slice separately
        vmin = min(r.min(), p.min())
        vmax = max(r.max(), p.max())
        nrows, ncols = _calc_plot_dim(p.shape[-1], f=0.5)
        fig = plt.figure(figsize=(4*ncols, 8.5*nrows))
        fig_grid = fig.add_gridspec(nrows, ncols, wspace=0.05, hspace=0.15,
            left=0.03, right=0.98, bottom=0.02, top=0.98)
        for iz in range(p.shape[-1]):
            ix = iz % ncols
            iy = iz // ncols
            ax1, ax2 = fig_grid[iy, ix].subgridspec(2, 1, hspace=0.03).subplots()
            ax1.imshow(p[:,:,iz].T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
            ax2.imshow(r[:,:,iz].T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
            ax1.axis('off')
            ax2.axis('off')
            ax1.set_title(f'z = {z_start + (iz + 0.5) * z_res:.2f}Å', fontsize=fontsize)
            if ix == 0:
                ax1.text(-0.1, 0.5, 'Prediction', ha='center', va='center',
                    transform=ax1.transAxes, rotation='vertical', fontsize=fontsize)
                ax2.text(-0.1, 0.5, 'Reference', ha='center', va='center',
                    transform=ax2.transAxes, rotation='vertical', fontsize=fontsize)

        plt.savefig(save_path:=os.path.join(outdir, f'{ind}_pred_dist.png'))
        if verbose > 0: print(f'Position distribution prediction image saved to {save_path}')
        plt.close()

        ind += 1

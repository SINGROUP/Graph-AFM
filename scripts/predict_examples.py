
import os
import sys
import time
import string
import random
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

sys.path.append('../ProbeParticleModel') # Make sure ProbeParticleModel is on PATH
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

sys.path.append('../src') # Add source code directory to Python PATH
import utils
import preprocessing as pp
from models import load_pretrained_model

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

classes     = [[1], [6, 14], [7, 15], [8, 16], [9, 17, 35]] # List of elements in each class
box_res     = (0.125, 0.125, 0.1)                           # Real-space voxel size for position distribution
zmin        = -0.8                                          # Maximum depth used for thresholding
peak_std    = 0.25                                          # Standard deviation of atom position peaks in angstroms
sequence_order = None                                       # Order for graph construction
device      = 'cuda'                                        # Device to run inference on
base_dir    = '../data'                                     # Base directory for molecule data
class_colors = ['w', 'dimgray', 'b', 'r', 'yellowgreen']    # Colors for classes
afm_slices  = [0, 5, 9]
marker_size = 25
z_min_marker = -1.0
z_max_marker = 0.0

def predict(model, molecules, box_borders):

    scan_dim = (
        int((box_borders[1][0] - box_borders[0][0]) / box_res[0]),
        int((box_borders[1][1] - box_borders[0][1]) / box_res[1]),
        20
    )
    print(scan_dim)
    afmulator_args = {
        'pixPerAngstrome'   : 20,
        'lvec'              : np.array([
                                [ 0.0,  0.0, 0.0],
                                [box_borders[1][0]+2.0,  0.0, 0.0],
                                [ 0.0, box_borders[1][1]+2.0, 0.0],
                                [ 0.0,  0.0, 6.0]
                                ]),
        'scan_dim'          : scan_dim,
        'scan_window'       : (box_borders[0][:2] + (7.0,), box_borders[1][:2] + (9.0,)),
        'amplitude'         : 1.0,
        'df_steps'          : 10,
        'initFF'            : True
    }

    generator_kwargs = {
        'batch_size'    : 30,
        'distAbove'     : 5.3,
        'iZPPs'         : [8],
        'Qs'            : [[ -10, 20,  -10, 0 ]],
        'QZs'           : [[ 0.1,  0, -0.1, 0 ]]
    }

    # Define AFMulator
    afmulator = AFMulator(**afmulator_args)
    afmulator.npbc = (0,0,0)

    # Define generator
    trainer = InverseAFMtrainer(afmulator, [], molecules, **generator_kwargs)

    # Generate batch    
    batch = next(iter(trainer))
    X, ref_graphs, ref_dist, box_borders = apply_preprocessing(batch, box_borders)

    with torch.no_grad():
        X_gpu = torch.from_numpy(X).unsqueeze(1).to(device)
        pred_graphs, pred_dist, pred_sequence, completed = model.predict_sequence(X_gpu, box_borders, order=sequence_order)
        pred_dist = pred_dist.cpu().numpy()

    return X, pred_graphs, pred_dist, ref_graphs, ref_dist, pred_sequence, box_borders

def apply_preprocessing(batch, box_borders):

    X, Y, atoms = batch

    pp.add_norm(X)
    pp.add_noise(X, c=0.1, randomize_amplitude=False)
    X = X[0]
    
    atoms = pp.top_atom_to_zero(atoms)
    bonds = utils.find_bonds(atoms)
    mols = [utils.MoleculeGraph(a, b, classes=classes) for a, b in zip(atoms, bonds)]
    mols = utils.threshold_atoms_bonds(mols, zmin)

    # shift_xy = [2, 0]
    # box_borders = (
    #     (box_borders[0][0] + shift_xy[0], box_borders[0][1] + shift_xy[1], box_borders[0][2]),
    #     (box_borders[1][0] + shift_xy[0], box_borders[1][1] + shift_xy[1], box_borders[1][2])
    # )
    # for m in mols:
    #     for a in m.atoms:
    #         a.xyz[0] += shift_xy[0]
    #         a.xyz[1] += shift_xy[1]

    ref_dist = utils.make_position_distribution([m.atoms for m in mols], box_borders,
        box_res=box_res, std=peak_std)
    
    return X, mols, ref_dist, box_borders

def get_marker_size(z, max_size=marker_size):
    return max_size * (z - z_min_marker) / (z_max_marker - z_min_marker)

def plot_xy(ax, mol, box_borders):

    if len(mol) > 0:

        mol_pos = mol.array(xyz=True)

        s = get_marker_size(mol_pos[:,2])
        if (s < 0).any():
            raise ValueError('Encountered atom z position(s) below box borders.')
        
        c = [class_colors[atom.class_index] for atom in mol.atoms]

        ax.scatter(mol_pos[:,0], mol_pos[:,1], c=c, s=s, edgecolors='k', zorder=2, linewidth=0.5)
        for b in mol.bonds:
            pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
            ax.plot(pos[:,0], pos[:,1], 'k', linewidth=1, zorder=1)
    
    ax.set_xlim(box_borders[0][0], box_borders[1][0])
    ax.set_ylim(box_borders[0][1], box_borders[1][1])
    ax.set_aspect('equal', 'box')

def plot_xz(ax, mol, box_borders):

    if len(mol) > 0:

        order = list(np.argsort(mol.array(xyz=True)[:, 1])[::-1])
        mol = mol.permute(order)
        mol_pos = mol.array(xyz=True)

        s = get_marker_size(mol_pos[:,2])
        if (s < 0).any():
            raise ValueError('Encountered atom z position(s) below box borders.')
        
        c = [class_colors[atom.class_index] for atom in mol.atoms]

        for b in mol.bonds:
            pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
            ax.plot(pos[:,0], pos[:,2], 'k', linewidth=1, zorder=1)
        ax.scatter(mol_pos[:,0], mol_pos[:,2], c=c, s=s, edgecolors='k', zorder=2, linewidth=0.5)
    
    ax.set_xlim(box_borders[0][0], box_borders[1][0])
    ax.set_ylim(box_borders[0][2], box_borders[1][2])
    ax.set_aspect('equal', 'box')

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

# Load model
model = load_pretrained_model('random', device=device)

# Make predictions
data = [
    predict(
        model,
        molecules=[os.path.join(base_dir, 'bcb.xyz')],
        box_borders=((2,2,-1.5),(18,18,0.5))
    ),
    predict(
        model,
        molecules=[os.path.join(base_dir, 'water.xyz')],
        box_borders=((2,2,-1.5),(18,18,0.5))
    ),
    predict(
        model,
        molecules=[os.path.join(base_dir, 'ptcda.xyz')],
        box_borders=((2,2,-1.5),(22,18,0.5))
    )
]
img_paths = [
    os.path.join(base_dir, 'bcb.png'),
    os.path.join(base_dir, 'water.png'),
    os.path.join(base_dir, 'ptcda.png')
]

# Initialize figure
width = 16.0 / 2.54
ns = len(afm_slices)
a = 1.2
h1, h2, h3 = 0.03 / 2.54, 0.2 / 2.54, 0.2 / 2.54
w1, w2, w3, w4, w5 = 0.1 / 2.54, 0.7 / 2.54, 0.75 / 2.54, 0.75 / 2.54, 0.2 / 2.54
y = (width - (w1 + w2 + w3 + w4 + w5) + (2*(1+a)*h1/ns + h2/2 + 32*h3/19)) / ((1+a)/ns + 1/2 + 32/19)
widths = [
    a*(y-2*h1)/ns,  # Molecule geometry
    w1,             # Padding
    (y-2*h1)/ns,    # AFM
    w2,             # Padding
    (y-h2)/2,       # Position grid
    w3,             # Padding
    16*(y-h3)/19,   # Predicted graph
    w4,             # Padding
    16*(y-h3)/19,   # Reference graph
    w5              # Padding
]
assert np.allclose(sum(widths), width)
heights = [y * d[0].shape[2] / d[0].shape[1] for d in data]
between_pad, bottom_pad, top_pad = 0.8 / 2.54, 0.7 / 2.54, 0.4 / 2.54
height_legend = 0.5 / 2.54
height = sum(heights) + (len(data) - 1) * between_pad + bottom_pad + top_pad + height_legend
fig = plt.figure(figsize=(width, height))

y0 = height - top_pad
for i, (X, pred_graphs, pred_dist, ref_graphs, ref_dist, pred_sequence, box_borders) in enumerate(data):

    extent = [box_borders[0][0], box_borders[1][0], box_borders[0][1], box_borders[1][1]]
    xticks = np.linspace(box_borders[0][0], box_borders[1][0], 5).astype(int)
    yticks = np.linspace(box_borders[0][1], box_borders[1][1], 5).astype(int)

    # Set subfigure reference letters
    fig.text(0.05/width, y0/height, string.ascii_uppercase[i], fontsize=10)

    # Create axes
    x = 0
    height_afm = (heights[i]-(ns-1)*h1)/ns
    height_grid = (heights[i]-h2)/2
    dy_graph = (heights[i]-h3)/19
    ax_img = fig.add_axes([x/width, (y0-heights[i])/height, widths[0]/width, heights[i]/height])
    x += widths[0] + widths[1]
    axes_afm = [fig.add_axes([x/width, (y0-height_afm*(j+1)-j*h1)/height, widths[2]/width,
        height_afm/height]) for j in range(ns)]
    x += widths[2] + widths[3]
    axes_grid2d = [
        fig.add_axes([x/width, (y0-height_grid)/height, widths[4]/width, height_grid/height]),
        fig.add_axes([x/width, (y0-heights[i])/height, widths[4]/width, height_grid/height])
    ]
    x += widths[4] + widths[5]
    axes_pred = [
        fig.add_axes([x/width, (y0-16*dy_graph)/height, widths[6]/width, 16*dy_graph/height]),
        fig.add_axes([x/width, (y0-heights[i])/height, widths[6]/width, 3*dy_graph/height])
    ]
    x += widths[6] + widths[7]
    axes_ref = [
        fig.add_axes([x/width, (y0-16*dy_graph)/height, widths[8]/width, 16*dy_graph/height]),
        fig.add_axes([x/width, (y0-heights[i])/height, widths[8]/width, 3*dy_graph/height])
    ]
    y0 -= heights[i] + between_pad


    # Plot molecule geometry
    xyz_img = np.flipud(imageio.imread(img_paths[i]))
    ax_img.imshow(xyz_img, origin='lower')
    ax_img.axis('off')

    # Plot AFM
    for s, ax in zip(afm_slices, axes_afm):
        ax.imshow(X[0][:, :, s].T, origin='lower', cmap='afmhot')
        ax.axis('off')

    # Plot grid in 2D
    p_mean, r_mean = pred_dist.mean(axis=-1), ref_dist.mean(axis=-1)
    vmin = min(r_mean.min(), p_mean.min())
    vmax = max(r_mean.max(), p_mean.max())
    for ax, d in zip(axes_grid2d, [p_mean, r_mean]):
        ax.imshow(d.T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax.tick_params('both', length=1, width=0.5, pad=1.5, labelsize=6)
        ax.spines[:].set_linewidth(0.3)
        ax.set_yticks(yticks)
    axes_grid2d[0].set_ylabel('Prediction, $y$(Å)', fontsize=6, labelpad=0)
    axes_grid2d[0].tick_params('x', bottom=False, labelbottom=False)
    axes_grid2d[1].set_xlabel('$x$(Å)', fontsize=6, labelpad=0)
    axes_grid2d[1].set_ylabel('Reference, $y$(Å)', fontsize=6, labelpad=0)
    axes_grid2d[1].set_xticks(xticks)

    # Plot graphs
    for axes, d in zip([axes_pred, axes_ref], [pred_graphs, ref_graphs]):
        plot_xy(axes[0], d[0], box_borders)
        plot_xz(axes[1], d[0], box_borders)
        axes[0].set_ylabel('$y$(Å)', fontsize=6, labelpad=0)
        axes[0].tick_params('both', length=1, width=0.5, pad=1, labelsize=6)
        axes[0].set_yticks(yticks)
        axes[0].tick_params('x', bottom=False, labelbottom=False)
        axes[0].spines[:].set_linewidth(0.3)
        axes[1].set_xlabel('$x$(Å)', fontsize=6, labelpad=0)
        axes[1].set_ylabel('$z$(Å)', fontsize=6, labelpad=0)
        axes[1].set_ylim(-2, 1)
        axes[1].tick_params('both', length=1, width=0.5, pad=1, labelsize=6)
        axes[1].set_xticks(xticks)
        axes[1].set_yticks([-2, -1, 0, 1])
        axes[1].spines[:].set_linewidth(0.3)

    if i == 0:
        axes_afm[0].set_title('AFM input', fontsize=9, pad=4)
        axes_grid2d[0].set_title('Position grid (2D)', fontsize=9, pad=4)
        axes_pred[0].set_title('Predicted graph', fontsize=9, pad=4)
        axes_ref[0].set_title('Reference graph', fontsize=9, pad=4)

# Add legend of classes and marker sizes
y0 += -between_pad + bottom_pad
ax_legend = fig.add_axes([0, 0, 1, height_legend/height])
ax_legend.axis('off')
ax_legend.set_xlim([0, 1])
ax_legend.set_ylim([0, 1])

y = 0.5
dx = [(len(c) + 1) * 0.022 for c in classes]
dx += [0.04, 0.092, 0.097, 0.097]
x = 0.5 - sum(dx)/2

# Class colors
for i, c in enumerate(classes):
    ax_legend.scatter(x, y, s=marker_size, c=class_colors[i], edgecolors='k', linewidth=0.5)
    t = ax_legend.text(x+0.01, y, ', '.join([utils.elements[e-1] for e in c]), fontsize=7,
        ha='left', va='center_baseline')
    x += dx[i]

# Marker sizes
x += dx[len(classes)]
marker_zs = np.array([z_max_marker, (z_min_marker + z_max_marker + 0.2) / 2, z_min_marker + 0.2])
ss = get_marker_size(marker_zs)
for i, (s, z) in enumerate(zip(ss, marker_zs)):
    ax_legend.scatter(x, y, s=s, c='w', edgecolors='k', linewidth=0.5)
    ax_legend.text(x + 0.01, y, f'z = {z}Å', fontsize=7, ha='left', va='center_baseline')
    x += dx[i+len(classes)+1]

plt.savefig('./predictions.pdf', dpi=300)
plt.close()

# 3D Distribution grids
fig = plt.figure(figsize=(16.0 / 2.54, 20 / 2.54))
fig_grid = fig.add_gridspec(len(data), 1, hspace=0.1, left=0.04, right=0.99, bottom=0.02, top=0.97,
    height_ratios=heights)
for i, (_, _, pred_dist, _, ref_dist, _, box_borders) in enumerate(data):
    p, r = pred_dist[0], ref_dist[0]
    z_start = box_borders[0][2]
    z_res = (box_borders[1][2] - box_borders[0][2]) / pred_dist.shape[-1]
    extent = [box_borders[0][0], box_borders[1][0], box_borders[0][1], box_borders[1][1]]
    vmin = min(r.min(), p.min())
    vmax = max(r.max(), p.max())
    nrows, ncols = 2, 10
    sample_grid = fig_grid[i].subgridspec(nrows, ncols, wspace=0.02, hspace=0.2/heights[i])
    for iz in range(p.shape[-1]):
        ix = iz % ncols
        iy = iz // ncols
        ax1, ax2 = sample_grid[iy, ix].subgridspec(2, 1, hspace=0.01).subplots()
        ax1.imshow(p[:,:,iz].T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax2.imshow(r[:,:,iz].T, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_title(f'z = {z_start + (iz + 0.5) * z_res:.2f}Å', fontsize=6, pad=2)
        if ix == 0:
            ax1.text(-0.1, 0.5, 'Prediction', ha='center', va='center',
                transform=ax1.transAxes, rotation='vertical', fontsize=6)
            ax2.text(-0.1, 0.5, 'Reference', ha='center', va='center',
                transform=ax2.transAxes, rotation='vertical', fontsize=6)
        if iz == 0:
            fig.text(-0.4, 1, string.ascii_uppercase[i], fontsize=10, transform=ax1.transAxes)
plt.savefig(f'grid3D.pdf', dpi=300)
plt.close()

# Plot prediction sequences
for i, (_, pred_graphs, _, _, _, pred_sequence, box_borders) in enumerate(data):
    seq = pred_sequence[0]
    atom_pos = pred_graphs[0].array(xyz=True)
    seq_len = len(seq) + 1
    x_seq = min(6, seq_len)
    y_seq = int(seq_len / (6+1e-6)) + 1
    fig = plt.figure(figsize=(16 / 2.54, (1.4 * heights[i] * y_seq + 0.4*(y_seq - 1)) / 2.54))
    grid_seq = fig.add_gridspec(y_seq, x_seq, wspace=0.02, hspace=0.3/heights[i], left=0.04, right=0.99,
        bottom=0.03/y_seq, top=1-0.1/y_seq)
    for j in range(len(seq) + 1):

        x_grid = j % x_seq
        y_grid = j // x_seq
        ax = fig.add_subplot(grid_seq[y_grid, x_grid])
        ax.spines[:].set_linewidth(0.3)

        s = get_marker_size(atom_pos[:,2], 8)
        ax.scatter(atom_pos[:,0], atom_pos[:,1], c='lightgray', s=s)

        if j > 0:
            mol, atom, bonds = seq[j-1]
            mol_pos = mol.array(xyz=True)
            atom_xyz = atom.array(xyz=True)
            c = class_colors[atom.class_index]
            s = get_marker_size(atom_xyz[2], 10)
            ax.scatter(atom_xyz[0], atom_xyz[1], c=c, s=s, edgecolors='k', zorder=2, linewidth=0.3)
            bonds = [i for i in range(len(bonds)) if bonds[i] > 0.5]
            for b in bonds:
                pos = np.vstack([mol_pos[b], atom_xyz])
                ax.plot(pos[:,0], pos[:,1], 'k', linewidth=0.6, zorder=1)
            if j > 1:
                s = get_marker_size(mol_pos[:,2], 10)
                c = [class_colors[atom.class_index] for atom in mol.atoms]
                ax.scatter(mol_pos[:,0], mol_pos[:,1], c=c, s=s, edgecolors='k', zorder=2, linewidth=0.3)
                for b in mol.bonds:
                    pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
                    ax.plot(pos[:,0], pos[:,1], 'k', linewidth=0.6, zorder=1)
        
        ax.set_xlim(box_borders[0][0], box_borders[1][0])
        ax.set_ylim(box_borders[0][1], box_borders[1][1])
        ax.set_aspect('equal', 'box')

        ax.set_title(f'{j}', fontsize=7, pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if j == 0:
            fig.text(-0.25, 1, string.ascii_uppercase[i], fontsize=10, transform=ax.transAxes)

    plt.savefig(f'pred_sequence_{i}.pdf', dpi=300)

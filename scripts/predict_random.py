
import os
import sys
import time
import glob
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

model_path  = './model.pth'                                 # Path to trained model weights
save_path   = './random_predictions.pdf'                    # File to save image in
classes     = [[1], [6, 14], [7, 15], [8, 16], [9, 17, 35]] # List of elements in each class
box_res     = (0.125, 0.125, 0.1)                           # Real-space voxel size for position distribution
zmin        = -0.8                                          # Maximum depth used for thresholding
peak_std    = 0.25                                          # Standard deviation of atom position peaks in angstroms
sequence_order = None                                       # Order for graph construction
device      = 'cuda'                                        # Device to run inference on
molecules_dir = './Molecules/'                              # Base directory for molecule data
class_colors = ['w', 'dimgray', 'b', 'r', 'yellowgreen']    # Colors for classes
marker_size  = 15
z_min_marker = -1.0
z_max_marker = 0.0

# Choose random molecules from the test set
num_mols = 10
molecules = [os.path.join(molecules_dir, f'test/{n}.xyz')
    for n in np.random.choice(range(35554), size=num_mols, replace=False)]
print(molecules)

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

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

# Download molecules if not already there
utils.download_molecules(molecules_dir, verbose=1)

# Load model
model = load_pretrained_model('random', device=device)

# Make predictions
X, pred_graphs, pred_dist, ref_graphs, ref_dist, pred_sequence, box_borders = predict(
    model, molecules=molecules, box_borders=((2,2,-1.5),(18,18,0.5)))

# Initialize figure
fig = plt.figure(figsize=(16 / 2.54, 16.5 / 2.54))
fig_grid = fig.add_gridspec(num_mols // 2, 2, wspace=0.07, hspace=0.22, top=0.97, bottom=0.03, left=0.01, right=0.99)

xticks = np.linspace(box_borders[0][0], box_borders[1][0], 5).astype(int)
yticks = np.linspace(box_borders[0][1], box_borders[1][1], 5).astype(int)

for i, (x, p, r) in enumerate(zip(X, pred_graphs, ref_graphs)):

    ix, iy = i % (num_mols // 2), 2 * i // num_mols
    sample_grid = fig_grid[ix, iy].subgridspec(1, 2, width_ratios=(1, 4), wspace=0.2)
    afm_axes = sample_grid[0, 0].subgridspec(2, 1, hspace=0.01).subplots()
    pred_ax, ref_ax = sample_grid[0, 1].subgridspec(1, 2, wspace=0.1).subplots()

    # Plot AFM
    for s, ax in zip([0, -1], afm_axes):
        ax.imshow(x[:, :, s].T, origin='lower', cmap='afmhot')
        ax.axis('off')

    # Plot graphs
    for ax, d in zip([pred_ax, ref_ax], [p, r]):
        plot_xy(ax, d, box_borders)
        ax.tick_params('both', length=1, width=0.5, pad=1, labelsize=6)
        ax.set_xticks(xticks)
        ax.set_xlabel('$x$(Å)', fontsize=6, labelpad=0)
        ax.spines[:].set_linewidth(0.3)
    pred_ax.set_yticks(yticks)
    pred_ax.set_ylabel('$y$(Å)', fontsize=6, labelpad=0)
    ref_ax.tick_params('y', left=False, labelleft=False)

    if (i % (num_mols // 2)) == 0:
        afm_axes[0].set_title('AFM sim.', fontsize=9, pad=4)
        pred_ax.set_title('Prediction', fontsize=9, pad=4)
        ref_ax.set_title('Reference', fontsize=9, pad=4)

plt.savefig(save_path, dpi=300)

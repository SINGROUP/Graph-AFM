

import os
import sys
import time
import glob
import h5py
import random
import numpy as np

sys.path.append('../ProbeParticleModel') # Make sure ProbeParticleModel is on PATH
from pyProbeParticle import oclUtils     as oclu
from pyProbeParticle import fieldOCL     as FFcl
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import AuxMap       as aux
from pyProbeParticle.AFMulatorOCL_Simple    import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2   import InverseAFMtrainer

sys.path.append('../src')
from utils import download_molecules

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)

def pad_xyzs(xyzs, max_len):
    xyzs_padded = [np.pad(xyz, ((0, max_len - len(xyz)), (0, 0))) for xyz in xyzs]
    xyzs = np.stack(xyzs_padded, axis=0)
    return xyzs

class Trainer(InverseAFMtrainer):

    # Override to randomize tip distance for each tip independently
    def handle_distance(self):
        self.randomize_distance(delta=0.25)
        self.randomize_tip(max_tilt=0.5)
        super().handle_distance()

# Options
molecules_dir   = './Molecules/'            # Where to save molecule database
save_path       = './graph_dataset.hdf5'    # Where to save training data

# Initialize OpenCL environment on GPU
env = oclu.OCLEnvironment( i_platform = 0 )
FFcl.init(env)
oclr.init(env)

afmulator_args = {
    'pixPerAngstrome'   : 20,
    'lvec'              : np.array([
                            [ 0.0,  0.0, 0.0],
                            [20.0,  0.0, 0.0],
                            [ 0.0, 20.0, 0.0],
                            [ 0.0,  0.0, 5.0]
                            ]),
    'scan_dim'          : (128, 128, 20),
    'scan_window'       : ((2.0, 2.0, 6.0), (18.0, 18.0, 8.0)),
    'amplitude'         : 1.0,
    'df_steps'          : 10,
    'initFF'            : True
}

generator_kwargs = {
    'batch_size'    : 30,
    'distAbove'     : 5.3,
    'iZPPs'         : [8],
    'Qs'            : [[ -10,  20,  -10, 0 ]],
    'QZs'           : [[ 0.1,   0, -0.1, 0 ]]
}

# Define AFMulator
afmulator = AFMulator(**afmulator_args)
afmulator.npbc = (0,0,0)

# Define AuxMaps
aux_maps = []

# Download molecules if not already there
download_molecules(molecules_dir, verbose=1)

# Paths to molecule xyz files
train_paths  = glob.glob(os.path.join(molecules_dir, 'train/*.xyz'))
val_paths    = glob.glob(os.path.join(molecules_dir, 'validation/*.xyz'))
test_paths   = glob.glob(os.path.join(molecules_dir, 'test/*.xyz'))

with h5py.File(save_path, 'w') as f:

    start_time = time.time()
    counter = 1
    total_len = np.floor((len(train_paths)+len(val_paths)+len(test_paths))/generator_kwargs['batch_size'])
    for mode, paths in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):

        # Define generator
        trainer = Trainer(afmulator, aux_maps, paths, **generator_kwargs)

        # Shuffle
        trainer.shuffle_molecules()

        # Calculate dataset shapes
        n_mol = len(trainer.molecules)
        max_mol_len = max([len(m) for m in trainer.molecules])
        X_shape = (
            n_mol,                                                  # Number of samples
            len(trainer.iZPPs),                                     # Number of tips
            afmulator.scan_dim[0],                                  # x size
            afmulator.scan_dim[1],                                  # y size
            afmulator.scan_dim[2] - afmulator.scanner.nDimConvOut   # z size
        )
        Y_shape = (n_mol, len(aux_maps)) + X_shape[2:4]
        xyz_shape = (n_mol, max_mol_len, 5)

        # Create new group in HDF5 file and add datasets to the group
        g = f.create_group(mode)
        X_h5 = g.create_dataset('X', shape=X_shape, chunks=(1,)+X_shape[1:], dtype='f')
        if len(aux_maps) > 0:
            Y_h5 = g.create_dataset('Y', shape=Y_shape, chunks=(1,)+Y_shape[1:], dtype='f')
        xyz_h5 = g.create_dataset('xyz', shape=xyz_shape, chunks=(1,)+xyz_shape[1:], dtype='f')

        # Generate data
        ind = 0
        for i, (X, Y, xyz) in enumerate(trainer):

            # Write batch to the HDF5 file
            n_batch = len(xyz)
            X_h5[ind:ind+n_batch] = np.stack(X, axis=1)
            if len(aux_maps) > 0:
                Y_h5[ind:ind+n_batch] = np.stack(Y, axis=1)
            xyz_h5[ind:ind+n_batch] = pad_xyzs(xyz, max_mol_len)
            ind += n_batch

            # Print progress info
            eta = (time.time() - start_time)/counter * (total_len - counter)
            print(f'Generated {mode} batch {i+1}/{len(trainer)} - ETA: {eta:.1f}s')
            counter += 1

print(f'Total time taken: {time.time() - start_time:.1f}s')

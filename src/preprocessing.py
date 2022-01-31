
import random
import numpy as np
import scipy.ndimage as nimg
from PIL import Image

def top_atom_to_zero(xyzs):
    '''
    Set the z coordinate of the highest atom in each molecule to 0.
    Arguments:
        xyzs: list of np.ndarray of shape (num_atoms, :). First three elements in axis 1 are xyz.
    Returns: new list of np.ndarrays of same shape as xyzs.
    '''
    new_xyzs = []
    for xyz in xyzs:
        xyz[:,2] -= xyz[:,2].max()
        new_xyzs.append(xyz)
    return new_xyzs

def add_noise(Xs, c=0.1, randomize_amplitude=False, normal_amplitude=False):
    '''
    Add uniform random noise to arrays. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, ...).
        c: float. Amplitude of noise. Is multiplied by (max-min) of sample.
        randomize_amplitude: Boolean. If True, noise amplitude is uniform random in [0,c]
                             for each sample in the batch.
        normal_amplitude: Boolean. If True and randomize_amplitude=True, then the noise amplitude
                          is distributed like the absolute value of a normally distributed variable
                          with zero mean and standard deviation equal to c.
    '''
    for X in Xs:
        sh = X.shape
        R = np.random.rand(*sh) - 0.5
        if randomize_amplitude:
            if normal_amplitude:
                amp = np.abs(np.random.normal(0, c, sh[0]))
            else:
                amp = np.random.uniform(0.0, 1.0, sh[0]) * c
        else:
            amp = [c] * sh[0]
        for j in range(sh[0]):
            X[j] += R[j] * amp[j]*(X[j].max()-X[j].min())

def add_norm(Xs, per_layer=True):
    '''
    Normalize arrays by subracting the mean and dividing by standard deviation. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, ...).
        per_layer: Boolean. If True, normalized separately for each element in last axis of Xs.
    '''
    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            if per_layer:
                for i in range(sh[-1]):
                    X[j,...,i] = (X[j,...,i] - np.mean(X[j,...,i])) / np.std(X[j,...,i])
            else:
                X[j] = (X[j] - np.mean(X[j])) / np.std(X[j])

def rand_shift_xy_trend(Xs, shift_step_max=0.02, max_shift_total=0.1):
    '''
    Randomly shift z layers in x and y. Each shift is relative to previous one. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, x_dim, y_dim, z_dim).
        shift_step_max: float in [0,1]. Maximum fraction of image size by which to shift for each step.
        max_shift_total: float in [0,1]. Maximum fraction of image size by which to shift in total.
    '''
    for X in Xs:
        sh= X.shape
        #calculate max possible shifts in pixexls between neighbor slices
        max_slice_shift_pix=np.floor(np.maximum(sh[1],sh[2])*shift_step_max).astype(int)
        #claculate max total shift in pixels 
        max_trend_pix = np.floor(np.maximum(sh[1],sh[2])*max_shift_total).astype(int)
        for j in range(sh[0]):
            rand_shift = np.zeros((sh[3],2))
            #calc values of random shift for slices in reverse order 
            # (0 values for closest slice) and biggest values for most far slices 
            for i in range(rand_shift.shape[0]-1, 0, -1):
                shift_values = [random.choice(np.arange(-max_slice_shift_pix,max_slice_shift_pix+1)),random.choice(np.arange(-max_slice_shift_pix,max_slice_shift_pix+1))]
                #print('shift_values = ', shift_values)
                for slice_ind in range(i): rand_shift[slice_ind,:] = rand_shift[slice_ind,:] + shift_values
                # cut shift values bigger than max_total_shift value    
                rand_shift = np.clip(rand_shift, -max_trend_pix,max_trend_pix).astype(int)
            for i in range(sh[3]):    
                shift_y = rand_shift[i,1]             
                shift_x = rand_shift[i,0]
                X[j,:,:,i] = nimg.shift (X[j,:,:,i], (shift_y,shift_x), mode='mirror' )   

def add_cutout(Xs, n_holes=5):
    '''
    Randomly add cutouts (square patches of zeros) to images. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, x_dim, y_dim, z_dim).
        n_holes: int. Maximum number of cutouts to add.
    '''
    
    def get_random_eraser(input_img,p=0.2, s_l=0.001, s_h=0.01, r_1=0.1, r_2=1./0.1, v_l=0, v_h=0):
        '''        
        p : the probability that random erasing is performed
        s_l, s_h : minimum / maximum proportion of erased area against input image
        r_1, r_2 : minimum / maximum aspect ratio of erased area
        v_l, v_h : minimum / maximum value for erased area
        '''

        sh = input_img.shape
        img_h, img_w = [sh[0], sh[1]] 
        
        if np.random.uniform(0, 1) > p:
            return input_img

        while True:
            
            s = np.exp(np.random.uniform(np.log(s_l), np.log(s_h))) * img_h * img_w
            r = np.exp(np.random.uniform(np.log(r_1), np.log(r_2)))
            
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        input_img[top:top + h, left:left + w] = 0.0

        return input_img

    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            for i in range(sh[3]):
                for attempt in range(n_holes):
                    X[j,:,:,i] = get_random_eraser(X[j,:,:,i])

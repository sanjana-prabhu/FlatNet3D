#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.io
import scipy.signal
import mat73
from tqdm import tqdm
from skimage import color, data, restoration
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage._shared.utils
import torch
import torch.fft as fft
from skimage.restoration import uft
# get_ipython().run_line_magic('matplotlib', 'inline')
from pypfm import PFMLoader
loader = PFMLoader(color=False, compress=False)


# !gdown --id 1aPrF1GU529pKyIA9LcoOSALUkH2JJZh5 # psfs
# !gdown --id 1l2og5PjTnhWWhdBXh8-LStlv6MM2PXge # flying things 3D subset(val)
# !gdown --id 1AbZ6e1VFDuEH3r2LPJtFKgzH7Vwp5ihw # flying things 3D subset(train and val)


# ### datasets

Flying_things3D = True
data_dir = '/media/data/salman/Lensless3D/'
data_dict_psf = mat73.loadmat(data_dir+'data/psfs_save_magfs.mat')
psf = data_dict_psf['psfs'][:,:,:,-25:][::2,::2]   
drng = data_dict_psf['drng'][-25:]
dataset_dir = data_dir + 'data/FlyingThings3D_subset/'
    
# ### helper functions

def show_figure(image1, title1, mode="single", image2=None, title2=None, save=False, img_name=None):
    
    if mode=='single':
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.2)

        im1 = ax.imshow(image1, cmap='gray')
        ax.set_title(title1)

        fig.colorbar(im1, cax=cax, orientation='vertical')
        
    elif mode=='comparison':
        fig, (ax1, ax2) = plt.subplots(1, 2)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.2)

        im1 = ax1.imshow(image1, cmap='gray')
        ax1.set_title(title1)

        fig.colorbar(im1, cax=cax, orientation='vertical')
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.2)

        im2 = ax2.imshow(image2, cmap='gray')
        ax2.set_title(title2)

        fig.colorbar(im2, cax=cax, orientation='vertical')
        fig.tight_layout(pad=1.0)
        fig.show()
        
    if save:
        fig.savefig(img_name)


from struct import *
def load_pfm(file_path):
    """
    load image in PFM type.
    Args:
        file_path string: file path(absolute)
    Returns:
        data (numpy.array): data of image in (Height, Width[, 3]) layout
        scale (float): scale of image
    """
    with open(file_path, encoding="ISO-8859-1") as fp:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        # load file header and grab channels, if is 'PF' 3 channels else 1 channel(gray scale)
        header = fp.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # grab image dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # grab image scale
        scale = float(fp.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # grab image data
        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        # reshape data to [Height, Width, Channels]
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data


def per_image_pipeline(img, new_depth, filename, device, noise_level,train):
    
    measurement = np.zeros([456, 684, 3])
    left = (data_dict_psf['psfs'].shape[0]//2-128)//2
    top = (data_dict_psf['psfs'].shape[1]//2-128)//2
    
    for c in range(3):
        img_scale = 1
        j = 0
        ch = c
        if c==2:
            ch = -1
        for i in np.unique(new_depth):
            img1 = img[:,:,c][::2,::2].copy()
            j+=1
            img1[np.where(new_depth != i)] = 0
            img1 = np.pad(img1,((left,left),(top,top)))/255
            measurement[:, :, c] += np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(data_dict_psf['psfs'][:,:,ch,24+int(i)][::2,::2])*np.fft.fft2(img1))))
        measurement[:, :, c] = measurement[:, :, c]/j
        
    max_m = np.max(measurement)
    mse = 10**(np.log10(max_m**2)-noise_level/10)
    noise = np.random.normal(0, np.sqrt(mse), np.shape(measurement))
    noisy_measurement = measurement + noise
    measurement = noisy_measurement
    measurement -= np.min(measurement)
    measurement /= np.max(measurement)
    measurement = (255*measurement).astype(np.uint8)
    
    if train:
        path2 = data_dir+'train_set/meas_rand_noise/'
    else:
        path2 = data_dir+'val_set/meas_rand_noise/'
        
    cv.imwrite(path2+filename+'.png', cv.cvtColor(measurement, cv.COLOR_RGB2BGR)) 
    
device = 'cuda:1'
 


# In[36]:


#np.random.randint(3)
mode1 = np.linspace(30,50)
mode0 = np.linspace(30,60)
def gen_noise_level():
    mode = np.random.randint(3)
    if mode == 0:
        return np.random.choice(mode0)
    return np.random.choice(mode1)

if Flying_things3D:
    train_size = 21818
    val_size = 4248
    for i in tqdm(range(train_size)):
        img_no = '0'*(7-len(str(i))) + str(i)
        noise_level = gen_noise_level()
        img = np.array(Image.open(dataset_dir+'train/left/'+str(img_no)+'.png'))[::2, ::2, :]
        depth = np.load('train_set/quan_depth/im'+str(img_no)+'.npy')
        per_image_pipeline(img[:256, 100:356, :], depth, 'im'+str(img_no), device, noise_level,True)        
        
    for i in tqdm(range(3000)):
        img_no = '0'*(7-len(str(i))) + str(i)
        noise_level = gen_noise_level()
        img = np.array(Image.open(dataset_dir+'val/left/'+str(img_no)+'.png'))[::2, ::2, :]
        depth = np.load('val_set/quan_depth/im'+str(img_no)+'.npy')
        per_image_pipeline(img[:256, 100:356, :], depth, 'im'+str(img_no), device, noise_level,False) 

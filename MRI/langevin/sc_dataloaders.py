from scipy.ndimage.interpolation import rotate,zoom
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import h5py
import sigpy as sp
from sc_utils import get_mvue
import pickle as pkl
from xml.etree import ElementTree as ET
import sys

class MVU_Estimator_NYU_Knees(Dataset):
    def __init__(self,input_dir,
                 project_dir='./',
                 R=8,
                 snr=10,
                 sigma_meas=0.07,
                 image_size=(256,256),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical'):
        # Attributes
        self.input_dir = input_dir
        self.project_dir = project_dir
        self.acs_size     = acs_size
        self.R = R
        self.snr = snr
        self.sigma_meas = sigma_meas
        self.image_size = image_size
        self.pattern      = pattern
        self.orientation  = orientation

    @property
    def num_slices(self):
        return 1

    @property
    def slice_mapper(self):
        return np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load specific slice from specific scan
        #gt_ksp = np.load(f'/shared/radon/MRI/sbhadra/pulse_null_functions/noisy_snr_{self.snr}_input_g_axial_60010/mask_random_{self.R}_fold_cartesian/axial_60010_0.npy') # obj 1
        gt_ksp = np.load(f'../noisy_sigma_{self.sigma_meas}_input_g_axial_60546/mask_random_{self.R}_fold_cartesian/axial_60546_0.npy') # obj 2
        mask = np.load(f'../masks/mask_random_{self.R}_fold_cartesian.npy')
        mask = np.fft.ifftshift(mask)
        # with h5py.File(os.path.join(self.project_dir, self.file_list[scan_idx]), 'r') as data:
        #     # Get kspace, mask
        #     gt_ksp = np.asarray(data['kspace'])[slice_idx]
        #     mask = np.asarray(data['masks'])[slice_idx]


        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape))

        # # Load MVUE slice from specific scan
        # mvue_file = os.path.join(self.input_dir,
        #                          os.path.basename(self.file_list[scan_idx]))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # apply mask
        #gt_ksp *= mask[None, :, :]
        #gt_ksp *= mask

        # Output
        sample = {
                  'mvue': mvue,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}
                  #'mvue_file': mvue_file}
        return sample


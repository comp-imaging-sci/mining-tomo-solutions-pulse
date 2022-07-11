import numpy as np
import PIL
import PIL.Image
import sys
import os
import glob
import scipy
import scipy.ndimage
#import dlib
#from drive import open_url
from pathlib import Path
import argparse
#from bicubic import BicubicDownSample
import torchvision
#from shape_predictor import align_face
#import numpy.fft as fft
from add_noise import add_noise
from torch_radon import RadonFanbeam
import torch

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='realpics', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input', help='output directory')
#parser.add_argument('-mask_type', type=str, default='mask_random_8_fold_cartesian_256x256', help='Type of mask')
parser.add_argument('-angle_factor', type=int, default=3, help='Divide 2*pi by angle factor for limited angle')
parser.add_argument('-loss_domain', type=str, choices=['image','meas'], default='image', help='Domain of least squares loss function')
parser.add_argument('-output_size', type=int, default=512, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-I', type=float, help='Incident X-ray intensity')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

#output_dir = Path(args.output_dir) / Path(args.mask_type) 
output_dir = Path(args.output_dir) / Path('af_'+str(args.angle_factor))
output_dir.mkdir(parents=True,exist_ok=True)

# Load the mask
#mask = np.load('./masks/'+args.mask_type+'.npy')

# Simulate the fan beam projection data
device = torch.device('cuda')
n_angles = 512 // args.angle_factor
fanbeam_angles = np.linspace(0, 2*np.pi/args.angle_factor, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)

# Add Poisson noise
def noisy_meas(g_bar,I):
    y_bar = I * torch.exp(-g_bar)
    y = torch.poisson(y_bar)
    return y

im_count = 0
for im in Path(args.input_dir).glob("*.*"):
    #faces = align_face(str(im),predictor)
    image = np.load(im)
    image = 1e-2 * image
    image_t = torch.FloatTensor(image).to(device)
    sinogram = radon_fanbeam.forward(image_t)
    y = noisy_meas(sinogram,args.I)
    y_np = y.detach().clone().cpu().numpy()
    np.save(Path(output_dir) / (im.stem+f"_{im_count}.npy"),y_np)
    im_count += 1


    # for i,face in enumerate(faces):
    #     if(args.output_size):
    #         factor = 1024//args.output_size
    #         assert args.output_size*factor == 1024
    #         D = BicubicDownSample(factor=factor)
    #         face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
    #         face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
    #         face = torchvision.transforms.ToPILImage()(face_tensor_lr)

    #     face.save(Path(args.output_dir) / (im.stem+f"_{i}.png"))

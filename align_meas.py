import numpy as np
import PIL
import PIL.Image
import sys
import os
import glob
import scipy
import scipy.ndimage
import dlib
from drive import open_url
from pathlib import Path
import argparse
from bicubic import BicubicDownSample
import torchvision
from shape_predictor import align_face
import numpy.fft as fft
from add_noise import add_noise 

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='realpics', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input', help='output directory')
parser.add_argument('-mask_type', type=str, default='mask_random_8_fold_cartesian_256x256', help='Type of mask')
parser.add_argument('-loss_domain', type=str, choices=['image','meas'], default='image', help='Domain of least squares loss function')
parser.add_argument('-output_size', type=int, default=256, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-SNR', type=float, help='SNR of measurement noise')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir) / Path(args.mask_type) 
output_dir.mkdir(parents=True,exist_ok=True)

# Load the mask
mask = np.load('./masks/'+args.mask_type+'.npy')

def f_meas(f,mask,SNR=None):
    mask = fft.ifftshift(mask)
    g = fft.fft2(f,norm='ortho')
    if SNR is not None:
        g = add_noise(g,SNR,mode='complex')
    g = mask * g
    f_meas = np.real(fft.ifft2(g,norm='ortho'))
    return f_meas.astype(np.float32),g.astype(np.complex64)

#print("Downloading Shape Predictor")
#f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
#f = 'shape_predictor_68_face_landmarks.dat'
#predictor = dlib.shape_predictor(f)

im_count = 0
for im in Path(args.input_dir).glob("*.*"):
    #faces = align_face(str(im),predictor)
    image = np.load(im)
    image,g = f_meas(image,mask,args.SNR)
    if args.loss_domain == 'image':
        np.save(Path(output_dir) / (im.stem+f"_{im_count}.npy"),image)
    else:
        np.save(Path(output_dir) / (im.stem+f"_{im_count}.npy"),g)
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

import numpy as np
import PIL
import PIL.Image
import sys
import os
import glob
import scipy.io as sio
import scipy.ndimage
from pathlib import Path
import argparse
import torchvision
import torch
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='realpics', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input', help='output directory')
#parser.add_argument('-mask_type', type=str, default='mask_random_8_fold_cartesian_256x256', help='Type of mask')
parser.add_argument('-n_angles', type=int, default=64, help='Number of views in the fanbeam projector')
parser.add_argument('-la', type=int, default=120, help='Limited angle range in the fanbeam projector')
parser.add_argument('-mu_max', type=float, default=0.0657, help='Maximum attenuation coefficient')
parser.add_argument('-loss_domain', type=str, choices=['image','meas'], default='image', help='Domain of least squares loss function')
parser.add_argument('-output_size', type=int, default=512, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-I', type=float, help='Incident X-ray intensity')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-limited_angle', action='store_true', help='Limited angle case')
parser.add_argument('-debug', action='store_true', help='Meas. data creation for debugging')

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

#output_dir = Path(args.output_dir) / Path(args.mask_type)
# if args.limited_angle and not(args.debug):
#     output_dir = Path(args.output_dir) / Path('la_'+str(args.la))
# elif args.debug:
#     output_dir = Path(args.output_dir) / Path('debug')
# else:
#     output_dir = Path(args.output_dir) / Path(str(args.n_angles)+'_views')

# output_dir.mkdir(parents=True,exist_ok=True)

if args.limited_angle:
    output_dir = Path(args.output_dir) / Path('la_'+str(args.la))
else:
    output_dir = Path(args.output_dir) / Path(str(args.n_angles)+'_views')

output_dir.mkdir(parents=True,exist_ok=True)


# Convert sparse system matrix to sparse tensor
def to_sparse_tensor(H_csc):
    H_coo = H_csc.tocoo()
    values = H_coo.data
    indices = np.vstack((H_coo.row,H_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = H_coo.shape
    H = torch.sparse_coo_tensor(i,v,shape)
    return H

# Add Poisson noise
def noisy_meas(g_bar,I):
    y_bar = I * np.exp(-g_bar)
    y = np.random.poisson(y_bar)
    #y = np.where(y<=0.1,0.1,y)
    #g = -np.log(y/I)
    return y

im_count = 0
for im in Path(args.input_dir).glob("*.*"):
    image = np.load(im)
    image = image.reshape(512**2,1)
    image_t = torch.FloatTensor(image).to('cuda')
    if args.limited_angle:
        H_csc = scipy.io.loadmat('H_la_'+str(args.la)+'.mat')['H']
    else:
        H_csc = scipy.io.loadmat('H_views_'+str(args.n_angles)+'.mat')['H']
    H = to_sparse_tensor(H_csc).to('cuda')
    sinogram_t = 0.82*args.mu_max*torch.sparse.mm(H,image_t)
    #print(sinogram_t.shape)
    sinogram = sinogram_t.cpu().numpy()
    if args.I is not None:
        y = noisy_meas(sinogram,args.I)
        np.save(Path(output_dir) / (im.stem+f"_{im_count}.npy"),y)
        #sio.savemat(Path(output_dir) / (im.stem+f"_{im_count}.mat"),{'g':g})
    else:
        np.save(Path(output_dir) / (im.stem+f"_{im_count}.npy"),sinogram)
    #plt.figure(); plt.imshow(-np.log(y/args.I).reshape(args.n_angles,768),aspect=20.0,cmap='gray'); plt.colorbar()
    im_count += 1

#plt.show()
    # for i,face in enumerate(faces):
    #     if(args.output_size):
    #         factor = 1024//args.output_size
    #         assert args.output_size*factor == 1024
    #         D = BicubicDownSample(factor=factor)
    #         face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
    #         face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
    #         face = torchvision.transforms.ToPILImage()(face_tensor_lr)

    #     face.save(Path(args.output_dir) / (im.stem+f"_{i}.png"))

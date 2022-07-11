import numpy as np 
import matplotlib.pyplot as plt 
import numpy.linalg as LA
import os, sys
sys.path.append('../')
import utils
import pickle
from PIL import Image
import torch
from torch_radon import RadonFanbeam

lr = '0.4'
lmda = '1e3' # geocross reg. parameter
n_angles = 25
fraction = '0.5'
img_dir = '/shared/radon/MRI/sbhadra/pulse_ct/runs/'+str(n_angles)+'_views/bn_lr_0.4_frac_0.5_geocross_'+lmda+'/'
img_idx = '0'
eps = 0.15
g = np.load('./input_g_'+img_idx+'/'+str(n_angles)+'_views/gt_'+img_idx+'_0.npy')
g_norm = LA.norm(g)
tol = eps * g_norm
save_dir = '/shared/radon/MRI/sbhadra/pulse_ct/runs/'+str(n_angles)+'_views/accepted_frac_'+fraction+'_eps_'+str(eps)+'_lr_'+lr+'_gc_'+lmda+'/'
save_img_dir = '/shared/radon/MRI/sbhadra/pulse_ct/runs/'+str(n_angles)+'_views/accepted_frac_'+fraction+'_eps_'+str(eps)+'_lr_'+lr+'_gc_'+lmda+'_images/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

device = torch.device('cuda')
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=2.5)

# Function for converting float32 image array to uint8 array in the range [0,255]
def convert_to_uint(img):
    img = 255 * (img-img.min())/(img.max()-img.min())
    return img.astype(np.uint8)

def residual_norm(g,f_hat):
    img = torch.FloatTensor(np.squeeze(f_hat)).to(device)
    sinogram = radon_fanbeam.forward(img)
    sinogram_np = sinogram.detach().clone().cpu().numpy()
    return LA.norm(g-sinogram_np)

folders = 0
for _, dirnames, _ in os.walk(img_dir):
    folders += len(dirnames)

#accepted_images = []
num_accept = 0; num_accept_restart = 0
num_reject = 0
count = 0

for idx in range(1,101):
    print('Starting restart '+str(idx)+'\n')
    residual_all = np.array([])
    for it in range(0,2000,100):
        print('It = '+str(it))
        subdir = img_dir + 'gt_'+img_idx+'_0_'+str(idx)+'/HR/'
        if it==0:
            img = np.load(subdir+'gt_'+img_idx+'_0_'+str(idx)+'_00.npy')
        else:
            img = np.load(subdir+'gt_'+img_idx+'_0_'+str(idx)+'_'+str(it)+'.npy')
        img = img * (img>0)
        residual = residual_norm(g,img)
        residual_all = np.append(residual_all,residual)
        print('Data fidelity fraction = '+str(residual/g_norm))
        if residual <= tol:
            print('Accept, residual = '+str(residual))
            np.save(save_dir+'img_'+str(num_accept)+'.npy',img)
            img = convert_to_uint(img)
            img = Image.fromarray(np.squeeze(img))
            img.save(save_img_dir+'img_'+str(num_accept)+'.png')
            num_accept +=1
        else:
            print('Reject, residual = '+str(residual))
            num_reject += 1
    if any(residual_all<=tol):
        num_accept_restart +=1
        print('Restart accepted')
    count += 1

        
print('Total number of accepted images = '+str(num_accept))
print('Total number of rejected images = '+str(num_reject))
rej_rate = 1 - num_accept_restart/count
print('Rejection rate = '+str(rej_rate))
np.save(save_dir+'rej_rate.npy',rej_rate)







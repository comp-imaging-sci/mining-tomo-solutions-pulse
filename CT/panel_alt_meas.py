# Analyze alternate images based on the minimum KL loss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import os, sys
import scipy.io as sio
import cupy as cp 
from cupy_airt_pinv import airt_pinv
import cupyx.scipy.sparse as cusp
sys.path.append('../')
matplotlib.rcParams.update({'font.size': 8})

servers = ['radon']
results_dir = './results_101421/'
#results_dir = './ct_figures/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Image parameters
img_idx = '1'
I = '1e3'
la = 1 # la = 1: limited angle, la = 0: limited views
n_proj = 120

# Optimization parameters
num_restarts = 100
lr = '0.4'
p = '0.01'
reg = '0.05'
eps = 46540.8 # Discrepancy tolerance
#eps = 50688.0 # Discrepancy tolerance

# Directory where alternate solutions are saved
for server in servers:
    if la:
        img_dir = '/shared/'+server+'/MRI/sbhadra/pulse_ct/runs/noisy_kl_I_'+I+'_airt_alpha_2/la_'+str(n_proj)+'/real_'+img_idx+'_lr_'+lr+'_p_'+p+'_reg_'+reg+'/'
        #img_dir = '/shared/'+server+'/MRI/sbhadra/pulse_ct/runs/noisy_kl_I_'+I+'_airt_alpha_3/la_'+str(n_proj)+'/real_'+img_idx+'_lr_'+lr+'_p_'+p+'_reg_'+reg+'/'
    else:
        img_dir = '/shared/'+server+'/MRI/sbhadra/pulse_ct/runs/noisy_kl_I_'+I+'_airt_alpha_2/'+str(n_proj)+'_views/real_'+img_idx+'_lr_'+lr+'_p_'+p+'_reg_'+reg+'/'
    if os.path.exists(img_dir):
        break

# Number of restarts completed
num_restarts = len(next(os.walk(img_dir))[1])
print('Number of restarts (completed) = '+str(num_restarts))

# Create subdirectory for saving results under each parameter setting
#save_dir = results_dir+'p_'+p+'_reg_'+reg+'/'
save_dir = results_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Load the true object and measurement data
idx = '1'
# gt = np.load('../real_'+img_idx+'/gt_'+img_idx+'.npy')
# gt = 0.01 * np.squeeze(gt)

#plt.figure(1); plt.imshow(gt,cmap='gray'); plt.colorbar(); plt.axis('off')
#plt.savefig(results_dir+'f_true.png',bbox_inches='tight')

# Sort alternate solutions based on the minimum KL loss
print('Sorting alternate solutions...')
min_kl_all = np.zeros(num_restarts)
num_rejected = 0

for restart in range(num_restarts):
    loss_kl = np.load(img_dir+'gt_'+idx+'_0_'+str(restart+1)+'/loss_kl.npy')
    #loss_kl = np.load(img_dir+'gt_0_'+str(restart+1)+'/loss_kl.npy')
    min_kl = np.min(loss_kl)
    min_kl_all[restart] = min_kl
    if min_kl>eps:
        num_rejected+=1

rejection_rate = num_rejected/num_restarts
print('Rejection rate = '+str(rejection_rate))
sorted_min_kl_all = np.sort(min_kl_all)
sorted_restarts = 1+np.argsort(min_kl_all)

# Extract the top 6 images and obtain corresponding measurement components
f_top = np.zeros((6,512,512),dtype=np.float32)

n_iter = 100000
H_sp = sio.loadmat('H_la_120.mat')['H']
H = cusp.csc_matrix(H_sp)

for top_i in range(6):
    f_np = np.load(img_dir+'gt_'+idx+'_0_'+str(sorted_restarts[top_i])+'/best_HR.npy')
    #f_np = np.load(img_dir+'gt_0_'+str(sorted_restarts[top_i])+'/best_HR.npy')
    f_np = np.squeeze(f_np)
    f_np = f_np * (f_np>0)
    recon = cp.asarray(f_np.reshape(512**2,1))
    Hf = H*recon
    print(f'i = {top_i}')
    f_meas_cp,_,_ = airt_pinv(Hf,H,n_iter)
    f_meas = cp.asnumpy(f_meas_cp)
    f_top[top_i,:,:] = f_meas

np.save(save_dir+'prev_meas.npy',f_top)

# fig1,ax1 = plt.subplots(2,3,constrained_layout=True)
# fig1.set_size_inches(12,8)

# for i in range(2):
#     for j in range(3):
#         idx = 3*i+j
#         img = f_top[idx]
#         im1 = ax1[i,j].imshow(img,cmap='gray'); ax1[i,j].axis('off'); im1.set_clim(-0.16,0.79)#; fig1.colorbar(im1,ax=ax1[i,j],fraction=0.046,pad=0.04); 

# fig1.savefig(save_dir+'curr_CT_alt_meas_I_'+I+'.png',dpi=300)

# plt.close('all')















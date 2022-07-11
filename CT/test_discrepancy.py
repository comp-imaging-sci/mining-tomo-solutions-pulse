# Test the discrepancy principle for Poisson noise model following Zanella et al. (2009)
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_radon import RadonFanbeam
import os

def noisy_meas(n_samples,g_bar,I):
    y_bar = I * np.exp(-g_bar)
    y = np.random.poisson(lam=y_bar,size=(n_samples,y_bar.shape[0],y_bar.shape[1]))
    return y

def data_fidelity(y,H_f,I): 
    df_constant = np.sum(y*np.log(y/I)-y,axis=(1,2)) 
    df = np.sum(y*H_f[np.newaxis,:,:],axis=(1,2)) + I * np.sum(np.exp(-H_f)) + df_constant 
    return df 

n_samples = 100000
device = 'cuda:0'
f = np.load('./real_1/gt_1.npy')
f = 1e-2 * np.squeeze(f)
f_t = torch.FloatTensor(f).to(device)

results_dir = './results_082121/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Study 1: Fix number of views, change maximum current strength
print('Fixed views = 512, change I_0...')
n_angles = 512
I_vals = [1e2,1e3,1e4,1e5]
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)
g_bar = radon_fanbeam.forward(f_t)
g_bar = g_bar.detach().cpu().numpy()
mean_df_array = np.array([],dtype=np.float32)

for I in I_vals:
    print('I_0 = '+str(I))
    # Mean of data fidelity across samples
    y = noisy_meas(n_samples,g_bar,I)
    df = data_fidelity(y,g_bar,I)
    mean_df = np.mean(df)
    print('Mean DF = '+str(mean_df))
    mean_df_array = np.append(mean_df_array,mean_df)
    # FBP from the first noisy data
    y0 = y[0,:,:].copy()
    g0 = -np.log(y0/I)
    g = torch.FloatTensor(g0).to(device)
    filtered_g = radon_fanbeam.filter_sinogram(g,filter_name='hann')
    fbp = radon_fanbeam.backprojection(filtered_g)
    fbp_np = fbp.detach().cpu().numpy()
    fbp_np = fbp_np * (fbp_np>0)
    plt.figure(); plt.imshow(fbp_np,cmap='gray');plt.colorbar();plt.axis('off')
    plt.savefig(results_dir+'fbp_views_512_I_'+str(I)+'.png',bbox_inches='tight')

np.save(results_dir+'mean_df_fixed_views_vary_I_2.npy',mean_df_array)

# Study 2: Fix max. current strength, change number of views
I = 1e4
print('\nFixed I_0 = 1e4, change number of views...')
n_angles_vals = [25,50,100,250]
mean_df_array = np.array([],dtype=np.float32)

for n_angles in n_angles_vals:
    fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)
    g_bar = radon_fanbeam.forward(f_t)
    g_bar = g_bar.detach().cpu().numpy()
    print('n_angles = '+str(n_angles))
    # Mean of data fidelity across samples
    y = noisy_meas(n_samples,g_bar,I)
    df = data_fidelity(y,g_bar,I)
    mean_df = np.mean(df)
    print('Mean DF = '+str(mean_df))
    mean_df_array = np.append(mean_df_array,mean_df)

    # FBP from the first noisy data
    y0 = y[0,:,:].copy()
    g0 = -np.log(y0/I)
    g = torch.FloatTensor(g0).to(device)
    filtered_g = radon_fanbeam.filter_sinogram(g,filter_name='hann')
    fbp = radon_fanbeam.backprojection(filtered_g)
    fbp_np = fbp.detach().cpu().numpy()
    fbp_np = fbp_np * (fbp_np>0)
    plt.figure(); plt.imshow(fbp_np,cmap='gray');plt.colorbar();plt.axis('off')
    plt.savefig(results_dir+'fbp_views_'+str(n_angles)+'_I_'+str(I)+'.png',bbox_inches='tight')

np.save(results_dir+'mean_df_fixed_I_vary_views_2.npy',mean_df_array)







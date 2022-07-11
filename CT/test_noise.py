# Model CT noise following Eq. (2) in the paper:
# Modeling mixed Poisson-Gaussian noise in statistical image reconstruction for X-ray CT by Ding et al.

import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch_radon import RadonFanbeam
plt.rcParams.update({'font.size': 12})

# Noise parameters
I = 1e3
k = 1
sigma = 0.0

# Simulate the fan beam projection data
n_angles = 512
device = torch.device('cuda')
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)

f = np.load('./real_1/gt_1.npy')
f = 1e-2 * np.squeeze(f)
f_t = torch.FloatTensor(f).to(device)
sinogram = radon_fanbeam.forward(f_t)
#g_bar = sinogram.detach().cpu().numpy()
g_bar = sinogram.clone()
y_bar = I * torch.exp(-g_bar)
y_bar_np = y_bar.detach().cpu().numpy()
print(y_bar_np.max())

y_e = sigma*torch.randn(y_bar.shape[0],y_bar.shape[1])
y = k * torch.poisson(y_bar) + torch.FloatTensor(y_e).to(device)
y_np = y.detach().cpu().numpy()

# Plot clean and noisy sinograms
#plt.figure();plt.imshow(g_bar_np,cmap='gray');plt.colorbar();plt.title('Clean sinogram')
#plt.figure();plt.imshow(g_np,cmap='gray');plt.colorbar();plt.title('Noisy sinogram')

# Plot clean and noisy FBP images
#g_bar_log = -torch.log(g_bar/I)
filtered_g_bar = radon_fanbeam.filter_sinogram(g_bar)
fbp_bar = radon_fanbeam.backprojection(filtered_g_bar)
fbp_bar_np = fbp_bar.detach().clone().cpu().numpy()
fbp_bar_np = fbp_bar_np * (fbp_bar_np>0)
#plt.figure();plt.imshow(fbp_bar_np,cmap='gray');plt.colorbar();plt.title('FBP from clean sinogram')

g_log = -torch.log(y/I) 
filtered_g = radon_fanbeam.filter_sinogram(g_log)
fbp = radon_fanbeam.backprojection(filtered_g)
fbp_np = fbp.detach().cpu().numpy()
fbp_np = fbp_np * (fbp_np>0)
#plt.figure();plt.imshow(fbp_np,cmap='gray');plt.colorbar();plt.title('FBP from noisy sinogram')
diff_im = np.abs(f-fbp_np)
#plt.show()

fig, ax = plt.subplots(2,3)
fig.set_size_inches(16,12)

#im0 = ax[0,0].imshow(f,cmap='gray'); fig.colorbar(im0,ax=ax[0,0],fraction=0.046,pad=0.04); ax[0,0].axis('off'); ax[0,0].set_title(r'$x$')
im0 = ax[0,0].imshow(y_bar_np.T,cmap='gray'); fig.colorbar(im0,ax=ax[0,0],fraction=0.046,pad=0.04); ax[0,0].set_xlabel('Viewing angle index'); ax[0,0].set_ylabel('Detector index'); ax[0,0].set_title(r'$\bar{y}$')
im1 = ax[1,0].imshow(f,cmap='gray'); fig.colorbar(im1,ax=ax[1,0],fraction=0.046,pad=0.04); ax[1,0].axis('off'); ax[1,0].set_title(r'$x$')
im2 = ax[0,1].imshow(y_np.T,cmap='gray'); fig.colorbar(im2,ax=ax[0,1],fraction=0.046,pad=0.04); ax[0,1].set_xlabel('Viewing angle index'); ax[0,1].set_ylabel('Detector index'); ax[0,1].set_title(r'$y$')
im3 = ax[1,1].imshow(fbp_np,cmap='gray'); fig.colorbar(im3,ax=ax[1,1],fraction=0.046,pad=0.04); ax[1,1].axis('off'); ax[1,1].set_title('FBP from noisy measurements')
im4 = ax[1,2].imshow(diff_im,cmap='gray'); fig.colorbar(im4,ax=ax[1,2],fraction=0.046,pad=0.04); ax[1,2].axis('off'); ax[1,2].set_title('Difference image')
ax[0,2].set_axis_off()

# fig, ax = plt.subplots(2,3)
# fig.set_size_inches(16,12)

# im0 = ax[0,0].imshow(f,cmap='gray'); fig.colorbar(im0,ax=ax[0,0],fraction=0.046,pad=0.04); ax[0,0].axis('off'); ax[0,0].set_title(r'$x$')
# im1 = ax[0,1].imshow(g_bar_np.T,cmap='gray'); fig.colorbar(im1,ax=ax[0,1],fraction=0.046,pad=0.04); ax[0,1].set_xlabel('Viewing angle index'); ax[0,1].set_ylabel('Detector index'); ax[0,1].set_title(r'$\bar{y}$')
# im2 = ax[0,2].imshow(g_np.T,cmap='gray'); fig.colorbar(im2,ax=ax[0,2],fraction=0.046,pad=0.04); ax[0,2].set_xlabel('Viewing angle index'); ax[0,2].set_ylabel('Detector index'); ax[0,2].set_title(r'$y$')
# im3 = ax[1,1].imshow(fbp_bar_np,cmap='gray'); fig.colorbar(im3,ax=ax[1,1],fraction=0.046,pad=0.04); ax[1,1].axis('off'); ax[1,1].set_title('FBP from 'r'$\bar{y}$')
# im4 = ax[1,2].imshow(fbp_np,cmap='gray'); fig.colorbar(im4,ax=ax[1,2],fraction=0.046,pad=0.04); ax[1,2].axis('off'); ax[1,2].set_title('FBP from 'r'$y$')
#ax[1,0].set_axis_off()

plt.figure()
plt.subplot(131); plt.imshow(f,cmap='gray');plt.colorbar(fraction=0.046,pad=0.04); plt.title('True object'); plt.axis('off')
plt.subplot(132); plt.imshow(fbp_np,cmap='gray');plt.colorbar(fraction=0.046,pad=0.04); plt.title('FBP from noisy \nprojections'); plt.axis('off')
plt.subplot(133); plt.imshow(diff_im,cmap='gray');plt.colorbar(fraction=0.046,pad=0.04); plt.title('Difference image'); plt.axis('off')
#fig.savefig('./examples_noisy/I_'+str(I)+'_k_'+str(k)+'_sigma_'+str(sigma)+'.png',dpi=300)

plt.show()



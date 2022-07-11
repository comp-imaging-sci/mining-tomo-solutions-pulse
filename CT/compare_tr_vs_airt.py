# Compare sinogram with fanbeam CT for torch-radon vs AirTools II
# CT system parameters as described in https://github.com/jakobsj/AIRToolsII/blob/master/demos/demo_astra_2d.m
# (512x512 instead of 128x128)
import numpy as np
import torch
from torch_radon import RadonFanbeam
import matplotlib.pyplot as plt
import numpy.linalg as la

device = 'cuda'
N = 512
n_angles = 360
det_count = int(np.ceil(np.sqrt(2)*N))
R = 2
dw = 2
sd = 3
source_distance = R*N 
det_distance = (sd-R)*N
det_spacing = dw*N/det_count
device = torch.device('cuda')
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(N,fanbeam_angles,source_distance=source_distance, det_distance=det_distance, det_count=det_count, det_spacing=det_spacing, clip_to_circle=True)

f = np.load('./real_1/gt_1.npy')
f = torch.FloatTensor(np.flipud(0.01*f).copy()).to(device)
sinogram_tr = radon_fanbeam.forward(f)
sinogram_tr = sinogram_tr.cpu().numpy()
#sinogram_tr = np.rot90(sinogram_tr,1)
#sinogram_tr = np.flipud(np.fliplr(sinogram_tr))
plt.figure(1); plt.imshow(sinogram_tr,cmap='gray',aspect=2.0); plt.colorbar();plt.title('torch-radon')

# Compare with AirTools
sinogram_at = np.load('sample_airt.npy')
sinogram_at = np.rot90(sinogram_at,k=1)
plt.figure(2); plt.imshow(sinogram_at,cmap='gray',aspect=2.0); plt.colorbar();plt.title('AirTools')

diff_sino = np.abs(sinogram_tr-sinogram_at)
plt.figure(3); plt.imshow(diff_sino,cmap='gray',aspect=2.0); plt.colorbar();plt.title('Difference of sinograms')
re = la.norm(diff_sino,'fro')/la.norm(sinogram_at,'fro')
print('Relative error = '+str(re))

sino_at_t = torch.FloatTensor(sinogram_at.copy()).to(device)
filtered_sino_at = radon_fanbeam.filter_sinogram(sino_at_t)
fbp_at = radon_fanbeam.backprojection(filtered_sino_at)
fbp_at = np.flipud(fbp_at.cpu().numpy())
fbp_at = fbp_at * (fbp_at>0)
plt.figure(4); plt.imshow(fbp_at,cmap='gray',aspect=1.0); plt.colorbar();plt.title('FBP from AirTools sinogram')

# sino_tr_t = torch.FloatTensor(sinogram_tr.T).to(device)
# filtered_sino_tr = radon_fanbeam.filter_sinogram(sino_tr_t)
# fbp_tr = radon_fanbeam.backprojection(filtered_sino_tr)
# fbp_tr = fbp_tr.cpu().numpy()
# plt.figure(5); plt.imshow(fbp_tr,cmap='gray',aspect=1.0); plt.colorbar();plt.title('FBP from torch-radon sinogram')
plt.show()


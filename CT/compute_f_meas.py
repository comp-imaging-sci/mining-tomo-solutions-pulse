# Compute the MP-pseudoinverse by obtaining the MNLS solution
import numpy as np
import torch
from torch_radon import RadonFanbeam
from torch_radon.solvers import Landweber
import matplotlib.pyplot as plt

def proj(x):
    return x

# Instantiate the fanbeam projector class
I = '1e3'
n_angles = 25
device = torch.device('cuda')
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)

# Load the true object and measurement data
idx = '1'
gt = np.load('./real_'+idx+'/gt_'+idx+'.npy')
gt = np.squeeze(gt)
g = np.load('./input_g_'+idx+'_I_'+I+'/'+str(n_angles)+'_views/gt_'+idx+'_0.npy')
g = torch.FloatTensor(g).to(device)

# Obtain the MNLS solution using Landweber iterations
#delta = 5.1 # Noise level for convergence based on discrepancy principle (real_0)
delta = 4.7 # Noise level for convergence based on discrepancy principle (real_1)
landweber_pinv = Landweber(radon_fanbeam,delta=delta,noiseless=False)
#landweber = Landweber(radon_fanbeam,delta=delta,projection=None)
alpha = 0.95 * landweber_pinv.estimate_alpha(512, device)
starting_point = torch.zeros((512,512),dtype=torch.float32).to(device)
reconstruction, progress = landweber_pinv.run(starting_point, g, alpha, iterations=100, callback=lambda xx: (torch.norm(radon_fanbeam.forward(xx)-g)**2).item())

f_pinv = reconstruction.detach().cpu().numpy()
#np.save('test_pinv_alpha_0.95.npy',f_pinv)

# Compute the measurement component of an alternate solution
landweber_meas = Landweber(radon_fanbeam)
#alpha = 0.95 * landweber_meas.estimate_alpha(512, device)
f_np = np.load('/shared/aristotle/MRI/sbhadra/pulse_ct/runs/noisy_kl_I_1e3_alpha_2/25_views/real_1_lr_0.4_p_0.9_reg_1e1/gt_'+idx+'_0_60/best_HR.npy')
f = torch.FloatTensor(f_np).to(device)
Hf = radon_fanbeam.forward(f)
reconstruction, progress = landweber_meas.run(starting_point, Hf, alpha, iterations=3800, callback=lambda xx: (torch.norm(radon_fanbeam.forward(xx)-Hf)**2).item())

f_meas = reconstruction.detach().cpu().numpy()
#np.save('test_f_meas_alpha_0.95.npy',f_meas)

plt.figure(1)
plt.subplot(121); plt.imshow(np.squeeze(gt),cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.axis('off'); plt.title('True object')
plt.subplot(122); plt.imshow(np.squeeze(f_pinv),cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.axis('off'); plt.title('Pseudoinverse')

f_np = f_np * (f_np>0)
plt.figure(2)
plt.subplot(121); plt.imshow(np.squeeze(f_np),cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.axis('off'); plt.title('Alternate solution')
plt.subplot(122); plt.imshow(np.squeeze(f_meas),cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.axis('off'); plt.title('Measurement component')

diff_meas = np.abs(f_meas-f_pinv)
plt.figure(3); plt.imshow(np.squeeze(diff_meas),cmap='gray'); plt.colorbar(); plt.axis('off')
plt.show()



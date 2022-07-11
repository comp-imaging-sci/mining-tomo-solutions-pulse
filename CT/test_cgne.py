# Compute the MP-pseudoinverse by obtaining the MNLS solution
import numpy as np
import torch
from torch_radon import RadonFanbeam
from torch_radon.solvers import cgne
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})

def proj(x):
    return x

# Instantiate the fanbeam projector class
I = '1e4'
n_angles = 25
device = torch.device('cuda')
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)

# Load the true object and measurement data
idx = '1'
gt = np.load('./real_'+idx+'/gt_'+idx+'.npy')
f_np = 0.01 * np.squeeze(gt)
f = torch.FloatTensor(f_np).to(device)
g = np.load('./input_g_'+idx+'_I_'+I+'/'+str(n_angles)+'_views/gt_'+idx+'_0.npy')
g = torch.FloatTensor(g).to(device)

# Obtain the MNLS solution using CGNE
Hf = radon_fanbeam.forward(f)
x_zero = torch.zeros((512,512),dtype=torch.float32).to(device)
f_meas_t, loss = cgne(radon_fanbeam,x_zero,Hf,callback=lambda xx: (torch.norm(radon_fanbeam.forward(xx)-Hf)**2).item(),max_iter=500,tol=1e-2)
f_meas = f_meas_t.detach().cpu().numpy()
f_null = f_np - f_meas
f_null_t = torch.FloatTensor(f_null).to(device)
Hf_null = radon_fanbeam.forward(f_null_t).cpu().numpy()

plt.figure(); plt.imshow(f_meas,cmap='gray'); plt.colorbar(); plt.title(r'$f_{meas}$')
plt.savefig('./results_cgne/f_meas_cgne.png',bbox_inches='tight')
plt.figure(); plt.imshow(f_null,cmap='gray'); plt.colorbar(); plt.title(r'$f_{null}$')
plt.savefig('./results_cgne/f_null_cgne.png',bbox_inches='tight')

# Map fwd of f_null
plt.figure(); plt.imshow(Hf_null,cmap='gray',aspect=10.0); plt.colorbar(); plt.title(r'$Hf_{null}$')
plt.savefig('./results_cgne/Hf_null_cgne.png',bbox_inches='tight')

# Difference between the forwards
Hf = Hf.cpu().numpy()
plt.figure(); plt.imshow(Hf,cmap='gray',aspect=10.0); plt.colorbar(); plt.title('Hf')
plt.savefig('./results_cgne/Hf.png',bbox_inches='tight')
Hf_meas = radon_fanbeam.forward(f_meas_t).cpu().numpy()
plt.figure(); plt.imshow(Hf_meas,cmap='gray',aspect=10.0); plt.colorbar(); plt.title(r'$Hf_{meas}$')
plt.savefig('./results_cgne/Hf_meas.png',bbox_inches='tight')
diff_fwd = np.abs(Hf-Hf_meas)
plt.figure(); plt.imshow(diff_fwd,cmap='gray',aspect=10.0); plt.colorbar()
plt.savefig('./results_cgne/diff_fwd.png',bbox_inches='tight')

plt.figure(); plt.plot(loss); plt.xlabel('Iterations'); plt.ylabel('Loss'); #plt.ylabel(r'$||g-Hf||^2$')
plt.savefig('./results_cgne/convergence_plot_cgne.png',bbox_inches='tight')
plt.figure(); plt.semilogy(loss); plt.xlabel('Iterations'); plt.ylabel('Loss') #plt.ylabel(r'$\log ||g-Hf||^2$')
plt.savefig('./results_cgne/semilog_convergence_plot_cgne.png',bbox_inches='tight')
# np.save('f_meas_cgne.npy',f_meas)
# np.save('loss_cgne.npy',loss)
plt.show()

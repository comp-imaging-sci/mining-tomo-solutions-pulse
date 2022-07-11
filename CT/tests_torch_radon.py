# Tests on the forward operator for correct implementation of measurement and null space projection operators
import numpy as np
import torch
from torch_radon import RadonFanbeam

# Perform the adjoint check
# <x^T,Hy> == <y^T,H^Tx>
n_angles = 100
device = torch.device('cuda')
fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)

f_x = np.load('./real_0/gt_0.npy')
f_x = torch.FloatTensor(f_x).to(device)
x_t = radon_fanbeam.forward(f_x)
x_t_bp = radon_fanbeam.backprojection(x_t)
x = x_t.cpu().numpy().astype(np.float64)
x_bp = x_t_bp.cpu().numpy().astype(np.float64)
y = np.load('./real_1/gt_1.npy')
y = torch.FloatTensor(y).to(device)
y_fwd = radon_fanbeam.forward(y).cpu().numpy()
y_fwd = y_fwd.astype(np.float64)
y = y.cpu().numpy().astype(np.float64)

adjoint_left = np.sum(x*y_fwd)
adjoint_right = np.sum(y*x_bp)
relative_error = np.abs(adjoint_left-adjoint_right)/adjoint_left

print('<x^T,Hy>='+str(adjoint_left))
print('<y^T,H^Tx>='+str(adjoint_right))
print('Relative error = '+str(relative_error))



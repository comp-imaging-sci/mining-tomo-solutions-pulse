import numpy as np
import cupy as cp
import scipy.io as sio
import cupy.linalg as cula
import cupyx.scipy.sparse as cusp
from cupy_airt_pinv import airt_pinv
import matplotlib.pyplot as plt

img_np = np.load('./real_2/gt.npy')
img = cp.asarray(img_np.reshape(512**2,1))
n_iter = 100000

g_np = np.load('test_g_rescaled_noisy.npy')
g = cp.asarray(g_np)
H_sp = sio.loadmat('H_la_120.mat')['H']
mu_max = 65.7*1e-3
px_size = 0.082
H = mu_max*px_size*cusp.csc_matrix(H_sp)
Hf = H*img

print('Computing pseudoinverse..')
f_pinv_cp,_,_ = airt_pinv(g,H,n_iter)
print('Computing meas. component..')
f_meas_cp,_,_ = airt_pinv(Hf,H,n_iter)
f_pinv = cp.asnumpy(f_pinv_cp)
f_meas = cp.asnumpy(f_meas_cp)

np.save('./cupy_test_rescaled/f_pinv.npy',f_pinv)
np.save('./cupy_test_rescaled/f_meas.npy',f_meas)




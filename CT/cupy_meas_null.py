import numpy as np
import cupy as cp
import scipy.io as sio
import cupy.linalg as cula
import cupyx.scipy.sparse as cusp
from cupy_airt_pinv import airt_pinv
import matplotlib.pyplot as plt

img_np = np.load('./fake_0/gt.npy')
img = cp.asarray(img_np.reshape(512**2,1))
n_iter = 100000

H_sp = sio.loadmat('H_la_120.mat')['H']
mu_max = 65.7*1e-3
px_size = 0.082
H = mu_max*px_size*cusp.csc_matrix(H_sp)
Hf = H*img

print('Computing meas. component..')
f_meas_cp,_,_ = airt_pinv(Hf,H,n_iter)
f_meas = cp.asnumpy(f_meas_cp)
f_null = img_np - f_meas

# np.save('./cupy_test_rescaled/f_pinv.npy',f_pinv)
# np.save('./cupy_test_rescaled/f_meas.npy',f_meas)
plt.figure(1);plt.imshow(img_np,cmap='gray');plt.colorbar();plt.axis('off')
plt.savefig('fake_0.png',bbox_inches='tight')
plt.figure(2);plt.imshow(f_meas,cmap='gray');plt.colorbar();plt.axis('off')
plt.savefig('fake_0_meas.png',bbox_inches='tight')
plt.figure(3);plt.imshow(f_null,cmap='gray');plt.colorbar();plt.axis('off')
plt.savefig('fake_0_null.png',bbox_inches='tight')

#plt.show()
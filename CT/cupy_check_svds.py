import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cula
import scipy.io as sio 
import scipy
print(cp.__version__)
cp.cuda.Device(3).use()

results_dir = './results_092921/'
# H = sio.loadmat('H_la_120.mat')['H']
# H = float(H)
# H_cu = cusp.csc_matrix(H)
# print(type(H_cu))
H_test = scipy.sparse.csc_matrix(scipy.random.rand(100,100))
#H_test = cp.random.rand(100,100)
H_csc = cusp.csc_matrix(H_test)
# print(H_cu.dtype)
print(H_csc.shape)
u,s,vt = cula.svds(H_csc,k=10,which='LM')
#u,s,vt = cula.svds(H_cu,k=np.min(H.shape)-1,which='LM')
print('Completed SVD computation')
u_npy = cp.asnumpy(u)
s_npy = cp.asnumpy(s)
vt_npy = cp.asnumpy(vt)
np.save(results_dir+'cu_u.npy',u_npy)
np.save(results_dir+'cu_s.npy',s_npy)
np.save(results_dir+'cu_vt.npy',vt_npy)

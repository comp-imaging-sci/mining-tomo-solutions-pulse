import numpy as np 
import cupy as cp 
import scipy.io as sio 
from cupy_airt_pinv import airt_pinv
import cupyx.scipy.sparse as cusp
import time

results_dir = './results_092821/'
num_recons = 6
mu_max = 0.01

n_iter = 100000
H_sp = sio.loadmat('H_la_120.mat')['H']

for i in range(num_recons):
    print(f'*** Recon {i} ***')
    recon_np = sio.loadmat(f'{results_dir}recon_{i}.mat')['recon']
    recon = cp.asarray(recon_np.reshape(512**2,1))
    H = cusp.csc_matrix(H_sp)
    Hf = H*recon
    start_time = time.time()
    f_meas_cp,_,_ = airt_pinv(Hf,H,n_iter)
    elapsed_time = time.time() - start_time
    print(f'Elapsed time = {elapsed_time} secs.')
    f_meas = cp.asnumpy(f_meas_cp)
    np.save(f'{results_dir}f_meas_{i}.npy',f_meas)

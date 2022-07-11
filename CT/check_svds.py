import numpy as np 
import scipy.io as sio 
from scipy.sparse.linalg import svds
import scipy
print(scipy.__version__)

results_dir = './results_092921/'
H = sio.loadmat('H_la_120.mat')['H']
H = H.astype(np.float32)
print(type(H))
#k = H.shape[1]-1
u,s,vt = svds(H,k=np.min(H.shape)-1,which='LM')
np.save(results_dir+'u.npy',u)
np.save(results_dir+'s.npy',s)
np.save(results_dir+'vt.npy',vt)

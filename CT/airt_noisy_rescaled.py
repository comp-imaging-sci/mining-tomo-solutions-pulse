import numpy as np
import scipy.io as sio
import torch
#from cupy_airt_pinv import airt_pinv
import matplotlib.pyplot as plt 

# Convert sparse system matrix to sparse tensor
def to_sparse_tensor(H_csc):
    H_coo = H_csc.tocoo()
    values = H_coo.data
    indices = np.vstack((H_coo.row,H_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = H_coo.shape
    H = torch.sparse_coo_tensor(i,v,shape)
    return H

# Add Poisson noise
def noisy_meas(g_bar,I):
    y_bar = I * np.exp(-g_bar)
    y = np.random.poisson(y_bar)
    y = np.where(y<=0.1,0.1,y)
    g = -np.log(y/I)
    return y,g

I = 1000
img = np.load('./real_2/gt.npy')
img = img.reshape(512**2,1)
img_t = torch.FloatTensor(img).to('cuda')
mu_max = 65.7*1e-3 # per mm
px_size = 0.82 # mm

H_csc = sio.loadmat('H_la_120.mat')['H']
H = to_sparse_tensor(H_csc).to('cuda')
H = mu_max * px_size * H
sinogram_t = torch.sparse.mm(H,img_t)
sinogram = sinogram_t.cpu().numpy()

y,g = noisy_meas(sinogram,I)
np.save('test_g_rescaled_noisy.npy',g)
y = y.reshape(120,768)
g = g.reshape(120,768)

plt.figure()
plt.subplot(131); plt.imshow(img.reshape(512,512),cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.title('True object')
plt.subplot(132); plt.imshow(y.T,cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.title('Photon count')
plt.subplot(133); plt.imshow(g.T,cmap='gray'); plt.colorbar(fraction=0.046,pad=0.04); plt.title('Sinogram')

#plt.show()






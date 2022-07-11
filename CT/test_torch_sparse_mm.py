import numpy as np 
import torch 
import scipy.io as sio
from airt_utils import to_sparse_tensor
import matplotlib.pyplot as plt 

f_np = np.load('./real_2/gt.npy')
f_np = f_np.reshape(512**2,1)
f = torch.FloatTensor(f_np).to('cuda')
scale_factor = 0.82*0.0657
H_csc = sio.loadmat('H_la_120.mat')['H']
H = scale_factor*to_sparse_tensor(H_csc).to('cuda')

I = 1
torch.manual_seed(100)
iters = 1
g_l1_all = torch.zeros(iters)
y_l1_all = torch.zeros(iters)
g_all = np.zeros((iters,768*120))
y_all = np.zeros((iters,768*120))

for i in range(iters):
    print('Iter = '+str(i))
    g = torch.sparse.mm(H,f)
    y = torch.exp(-g)
    g_all[i,:] = torch.squeeze(g).detach().cpu().numpy()
    y_all[i,:] = torch.squeeze(y).detach().cpu().numpy()
    g_l1_all[i] = torch.sum(torch.abs(g))
    y_l1_all[i] = torch.sum(torch.abs(y))

g_l1_all_np = g_l1_all.detach().cpu().numpy()
y_l1_all_np = y_l1_all.detach().cpu().numpy()

g_comp_std = np.std(g_all,axis=0)
y_comp_std = np.std(y_all,axis=0)

print(f'g_norm = {g_l1_all_np}')
print(f'y_norm = {y_l1_all_np}')

#plt.figure(1); plt.plot(g_l1_all_np); plt.title('l1 of g')
#plt.figure(2); plt.plot(y_l1_all_np); plt.title('l1 of y')

# plt.figure(1); plt.plot(g_comp_std); plt.title('g comp std')
# plt.figure(2); plt.plot(y_comp_std); plt.title('y comp std')

plt.show()














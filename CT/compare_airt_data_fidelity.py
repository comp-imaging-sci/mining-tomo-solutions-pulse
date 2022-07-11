import numpy as np 
import scipy.io as sio 

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
    return y

H = sio.loadmat('H_la_120.mat')['H']
f = np.load('./real_1/gt.npy').reshape(512**2,1)
I = 1e3

# Function to compute the part of data fidelity that is a function of f
def data_fidelity_varying(g_bar,y):
    df = np.sum(y*g_bar)+I*np.sum(np.exp(-g_bar))
    return df

# Previous scale
g_bar = 0.01*H*f
y = noisy_meas(g_bar,I)
df = data_fidelity_varying(g_bar,y)
print(f'Varying data fidelity (previous) = {df}')

# Current scale
H = 0.82*0.0657*H
g_bar = H*f
y = noisy_meas(g_bar,I)
df = data_fidelity_varying(g_bar,y)
print(f'Varying data fidelity (current) = {df}')

# Total data fidelity for current scale
df_total = df - 41692126.65
print(f'Bound based on true object = {df_total}')



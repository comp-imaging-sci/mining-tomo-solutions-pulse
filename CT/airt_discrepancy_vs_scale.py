import numpy as np 
import scipy.io as sio 
import cupy as cp 
import cupyx.scipy.sparse as cusp
import matplotlib.pyplot as plt

def add_noise(y):
    y_n = cp.random.poisson(y)
    return y_n

def df(y,g_bar):
    df_constant_i = cp.where(y>0,y*cp.log(y/I)-y,0)
    df_constant = cp.sum(df_constant_i)
    df_total = cp.sum(y*g_bar)+I*cp.sum(np.exp(-g_bar))+df_constant
    return df_total

I = 1000
iters = 10000
H_sp = sio.loadmat('H_la_120.mat')['H']
H = cusp.csc_matrix(H_sp)
f_np = np.load('./real_1/gt.npy')
f = cp.asarray(f_np.reshape(512**2,1))

# Compute expected data fidelity with scale 1
scale_1 = 0.01
g_bar_1 = scale_1*H*f
y_1 = I*cp.exp(-g_bar_1)
data_fidelity_1 = 0

scale_2 = 0.82*0.063
g_bar_2 = scale_2*H*f
y_2 = I*cp.exp(-g_bar_2)
data_fidelity_2 = 0

for i in range(iters):
    print(f'Iter = {i}')
    yn_1 = add_noise(y_1)
    data_fidelity_1 += df(yn_1,g_bar_1)
    yn_2 = add_noise(y_2)
    data_fidelity_2 += df(yn_2,g_bar_2)

print('exp df_1 = '+str(data_fidelity_1/iters))
print('exp df_2 = '+str(data_fidelity_2/iters))

plt.figure(1); plt.hist(yn_1.get(),bins=100,density=True); plt.title('Previous scale')
plt.savefig('./results_101421/hist_prev_scale.png',bbox_inches='tight')
plt.figure(2); plt.hist(yn_2.get(),bins=100,density=True); plt.title('Current scale')
plt.savefig('./results_101421/hist_curr_scale.png',bbox_inches='tight')
plt.show()




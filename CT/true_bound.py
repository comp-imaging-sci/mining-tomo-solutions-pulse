import numpy as np 
import scipy.io as sio

def df(y,g_bar,I):
    df_constant_i = np.where(y>0,y*np.log(y/I)-y,0)
    df_constant = np.sum(df_constant_i)
    df_total = np.sum(y*g_bar)+I*np.sum(np.exp(-g_bar))+df_constant
    return df_total, df_constant

idx = 3
I = 1e5
#I = 'airt_fbp_30_I_1e3.npy'
H = sio.loadmat('H_la_120.mat')['H']
#f = np.load('./real_1/gt.npy')
#f = np.load('./fake_0/gt.npy')
#f = np.load('./embed_1_0/gt.npy')
f = np.load(f'./embed_{idx}_0/gt.npy')
f = f.reshape(512**2,1)
#y = np.load('./input_airt_y_fake_0_I_1e3/la_120/gt_0.npy')
y = np.load(f'./input_airt_y_{idx}_I_1e5/la_120/gt_0.npy')
#y = np.load('./input_airt_y_embed_1_1_I_1e3/la_120/gt_0.npy')

if idx==1:
    mu_max=0.063
elif idx==3:
    mu_max=0.046

g_bar = 0.82*mu_max*H*f
df_true, df_constant = df(y,g_bar,I)
print('True bound = '+str(df_true))
print('df_constant = '+str(df_constant))

ratio_bound = 2*df_true/y.size
print('Ratio of bound = '+str(ratio_bound))

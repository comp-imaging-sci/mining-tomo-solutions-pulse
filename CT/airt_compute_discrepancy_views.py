import numpy as np 

I_order = 3
I = 10**I_order
n_angles = 24
M = 768*n_angles

y = np.load('./input_airt_y_1_I_1e'+str(I_order)+'/'+str(n_angles)+'_views/gt_1_0.npy')
df_constant = np.sum(y*np.log(y/I)-y)
print('df_constant = '+str(df_constant))

delta = 1.01*M/2
print('Discrepancy limit = '+str(delta))
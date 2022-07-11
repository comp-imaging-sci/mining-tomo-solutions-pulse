import numpy as np 

I_order = 3
#I = 10**I_order
I = 1e5
la = 120
M = 768*la

#y = np.load('./input_airt_y_2_I_1e'+str(I_order)+'/la_'+str(la)+'/gt_0.npy')
y = np.load('./input_airt_y_1_I_1e5/la_'+str(la)+'/gt_0.npy')
#y = np.load('./input_airt_y_fake_0_I_1e3/la_'+str(la)+'/gt_0.npy')
#y = np.load('./input_airt_y_1_I_1e3/debug/gt_0.npy')
df_constant_i = np.where(y>0,y*np.log(y/I)-y,0)
df_constant = np.sum(df_constant_i)
print('df_constant = '+str(df_constant))

delta = 1.1*M/2
print('Discrepancy limit = '+str(delta))
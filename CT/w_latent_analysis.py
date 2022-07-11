import numpy as np 
import matplotlib.pyplot as plt 
import numpy.linalg as LA
from scipy.spatial.distance import cdist
import os

results_dir = './latent_analysis_061721/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

latents_w = np.load('latents_w.npy')
latents_v = np.load('latents_v.npy')

mean_w = np.mean(latents_w,axis=0)
mean_v = np.mean(latents_v,axis=0)

print(latents_w.shape); print(mean_v.shape)

mean_w = mean_w[np.newaxis,:]
mean_v = mean_v[np.newaxis,:]

v_from_mean = latents_v - mean_v
#v_from_mean_norm = LA.norm(v_from_mean)**2
v_from_mean_norm = np.sum(v_from_mean**2,axis=1)
print(v_from_mean_norm.shape)
print(v_from_mean_norm.max())

plt.figure();plt.hist(v_from_mean_norm,bins=1000,density=True)
plt.savefig(results_dir+'hist_v_norm.png',bbox_inches='tight')

# Estimate covariance matrix and perform eigen decomposition
print('Estimating covariance matrix...')
Sigma = np.cov(latents_v,rowvar=False)
print('Shape of Sigma = '+str(Sigma.shape))

print('Performing eigen decomposition...')
Lambda, C = LA.eig(Sigma)
print('Min. eigen value ='+str(Lambda.min()))
Lambda_inv = 1/Lambda
Lambda_inv = np.diag(Lambda_inv)
Sigma_inv = np.matmul(C,np.matmul(Lambda_inv,C.T))

print('Computing Mahalanobis distance...')
v_md = cdist(latents_v,mean_v,'mahalanobis',VI=Sigma_inv)
print(v_md.shape)

plt.figure();plt.imshow(Sigma,cmap='gray');plt.colorbar()
plt.savefig(results_dir+'Sigma.png',bbox_inches='tight')

plt.figure(); plt.plot(Lambda)
plt.savefig(results_dir+'eigen_spectrum.png')

plt.figure();plt.hist(v_md,bins=1000,density=True)
plt.savefig(results_dir+'hist_v_md.png',bbox_inches='tight')

# With PULSE diagonal approximation of covariance matrix
var_w = np.var(latents_v,axis=0)
Sigma_pulse_inv = np.diag(1/var_w)
v_md_pulse = cdist(latents_v,mean_v,'mahalanobis',VI=Sigma_pulse_inv)

plt.figure();plt.hist(v_md_pulse,bins=1000,density=True)
plt.savefig(results_dir+'hist_v_md_pulse.png',bbox_inches='tight')







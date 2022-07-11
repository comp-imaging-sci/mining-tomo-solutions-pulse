import numpy as np 
import numpy.fft as fft
import numpy.linalg as LA 

def f_meas_null(f,mask):
    mask = fft.ifftshift(mask)
    g = mask*fft.fft2(f,norm='ortho')
    f_meas = np.real(fft.ifft2(g,norm='ortho'))
    f_null = f - f_meas
    return f_meas,f_null

def data_fidelity(f_meas,img_meas):
    return LA.norm(f_meas-img_meas,ord='fro')**2 / LA.norm(f_meas,'fro')**2

def residual_norm(g,f_hat,mask):
    mask = fft.ifftshift(mask)
    return LA.norm(g-mask*fft.fft2(f_hat,norm='ortho'))**2
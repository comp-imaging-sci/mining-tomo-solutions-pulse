import numpy as np
import numpy.linalg as la

def add_noise(x, SNR, mode='gaussian'):
    """ Adds gaussian noise of a given SNR to a signal
    """
    p_signal = la.norm(x)**2
    snr_inv = 10**(-0.1*SNR)
    p_noise = p_signal * snr_inv
    sigma = np.sqrt( p_noise/np.prod(x.shape) )
    if mode=='gaussian':
        x_noisy = x + sigma * np.random.randn(*(x.shape))
    elif mode=='salt_pepper':
        x_noisy = x + sigma * abs(np.random.randn(*(x.shape)))
    elif mode=='complex':
        x_noisy = x + sigma/np.sqrt(2) * (np.random.randn(*(x.shape)) + 1.j*np.random.randn(*(x.shape)))
    else:
        raise ValueError("Enter a suitable mode")
    return x_noisy.astype(x.dtype)

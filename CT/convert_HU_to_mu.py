import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

#filename = '/shared/curie/SOMS/sbhadra/NIH_CT_images/Images_png/000294_03_02/177.png' # real 1
#filename = '/shared/curie/SOMS/sbhadra/NIH_CT_images/Images_png/000294_03_02/190.png' # real 2
filename = '/shared/curie/SOMS/sbhadra/NIH_CT_images/Images_png/003842_01_01/086.png' # real 3
HU = np.array(Image.open(filename))
HU = HU - 32768

mu_water = 20
mu_air = 0.02
mu = 1e-3*HU*(mu_water-mu_air)+mu_water
mu_norm = mu/mu.max()
print(f'mu_max = {mu.max()}')

#np.save('./real_2/gt.npy',mu_norm)
#plt.figure(); plt.imshow(mu_norm,cmap='gray'); plt.colorbar(); plt.show()





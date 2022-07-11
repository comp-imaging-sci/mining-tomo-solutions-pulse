# Create separate folders for computing FID images from 20k images
import numpy as np 
from PIL import Image
import os

# Function for converting float32 image array to uint8 array in the range [0,255]
def convert_to_uint(img):
    img = 255 * (img-img.min())/(img.max()-img.min())
    return img.astype(np.uint8)

random_range = np.random.permutation(60000)

img_dir = '/shared/einstein/SOMS/sbhadra/NIH_CT_images/'
real_dir = '/shared/einstein/MRI/sbhadra/pulse_ct/real_dir/'
if not os.path.exists(real_dir):
    os.makedirs(real_dir)

count = 0
for i in random_range:
    img = np.load(f'{img_dir}img_{i}.npy')
    if (not img.shape[0]==512) or (not img.shape[0]==img.shape[1]):
        print(f'Shape not correct:{img.shape}')
        continue
    elif img.max() > 0:
        img_uint = convert_to_uint(img)
        im = Image.fromarray(img_uint)
        im.save(real_dir+'img_'+str(i)+'.png')
        count += 1
        print(f'Saving real image {count}')
    if count == 20000:
        break






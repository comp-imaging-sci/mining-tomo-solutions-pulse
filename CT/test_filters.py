import numpy as np 
import torch
from torch_radon import RadonFanbeam
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

device = 'cuda'
filter_types = ['ramp','shepp-logan','cosine','hamming','hann']

# Load the noisy measurement data
n_angles = 25
#g = np.load('./input_g_1_I_1e3/25_views/gt_1_0.npy')
g = np.load('./input_g_1/25_views/gt_1_0.npy')
g = torch.FloatTensor(g).to(device)

# Load the true object
f = np.load('/shared/radon/MRI/sbhadra/pulse_ct/real_1/gt_1.npy')
f = 1e-2 * np.squeeze(f)

fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)

for i in range(5):
    filter_name = filter_types[i]
    print('\nFilter name: '+filter_name)
    filtered_sinogram = radon_fanbeam.filter_sinogram(g,filter_name=filter_name)
    fbp = radon_fanbeam.backprojection(filtered_sinogram)
    fbp_np = fbp.detach().cpu().numpy()
    fbp_np = fbp_np * (fbp_np>0)
    diff_im = np.abs(f-fbp_np)
    ssim_value = ssim(f,fbp_np,data_range=fbp_np.max()-fbp_np.min())
    print('SSIM = '+str(ssim_value))
    rmse = np.sqrt(np.sum(diff_im**2))/512
    print('RMSE = '+str(rmse))
    plt.figure()
    plt.subplot(121); plt.imshow(fbp_np,cmap='gray');plt.colorbar();plt.title(filter_name)
    plt.subplot(122); plt.imshow(f,cmap='gray');plt.colorbar();plt.title('Difference image')

plt.show()
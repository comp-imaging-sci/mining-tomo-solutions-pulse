# Perform adjoint test using astra
# See /torch-radon/tests/test_fanbeam.py
import numpy as np 
import astra

def circle_mask(size, radius):
    center = (size - 1) / 2
    c0, c1 = np.ogrid[0:size, 0:size]
    return ((c0 - center) ** 2 + (c1 - center) ** 2) <= radius ** 2

clip_to_circle = False
spacing = 3.0
n_angles = 25
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
image_size = 512
det_count = 512
source_distance = 512
det_distance = 512
mask_radius = det_count/2.0 if clip_to_circle else -1

vol_geom = astra.create_vol_geom(512, 512)
proj_geom = astra.create_proj_geom('fanflat', spacing, det_count, angles, source_distance, det_distance)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

f_x = np.load('./real_0/gt_0.npy').astype(np.float64)
if clip_to_circle:
    f_x *= circle_mask(image_size, mask_radius)
idx, x = astra.create_sino(f_x,proj_id)
_, x_bp = astra.create_backprojection(x.astype(np.float64),proj_id)
if clip_to_circle:
    x_bp *= circle_mask(image_size, mask_radius)

y = np.load('./real_1/gt_1.npy')
_, y_fwd = astra.create_sino(y,proj_id)

adjoint_left = np.sum(x*y_fwd)
adjoint_right = np.sum(y*x_bp)

print('<x^T,Hy>='+str(adjoint_left))
print('<y^T,H^Tx>='+str(adjoint_right))
print('Relative error = '+str((np.abs(adjoint_right-adjoint_left))/adjoint_left))

astra.data2d.delete(proj_id)









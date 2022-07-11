from matplotlib import pyplot
import torch
import torch.nn as nn
import torchvision
import collections
from collections import OrderedDict
import stylegan
from stylegan import G_mapping, G_synthesis
import numpy as np

model_name = 'NIH_CT_5000'
mapping = G_mapping().cuda()
mapping.load_state_dict(torch.load('./pretrained_networks/'+model_name+'_mapping.pt'))

with torch.no_grad():
    torch.manual_seed(0)
    latents_z = torch.randn((100000,512),dtype=torch.float32, device="cuda")
    latents_w = mapping(latents_z)
    latents_v = torch.nn.LeakyReLU(5)(latents_w)
    latents_z_np = latents_z.detach().cpu().numpy()
    latents_w_np = latents_w.detach().cpu().numpy()
    latents_v_np = latents_v.detach().cpu().numpy()
    print(latents_w_np.shape)
    np.save('latents_z.npy',latents_z_np)
    np.save('latents_w.npy',latents_w_np)
    np.save('latents_v.npy',latents_v_np)




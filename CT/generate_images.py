# Referene: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

from matplotlib import pyplot
import torch
import torch.nn as nn
import torchvision
import collections
from collections import OrderedDict
import stylegan
from stylegan import G_mapping, G_synthesis
import numpy as np

dim = 512
num_w_layers = 16
#model_name = 'MRI_axial_256x256_norm'
model_name = 'NIH_CT_60000'

g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis())    
]))

#g_all.load_state_dict(torch.load('./pretrained_networks/'+model_name+'.for_g_all.pt'))
g_mapping = G_mapping().cuda()
g_synthesis = G_synthesis().cuda()
#print(g_synthesis.state_dict())

# print('Names of trainable parameters')
# for name,param in g_synthesis.named_parameters():
#     # if param.requires_grad:
#     #     print(name)
#     print(name)

g_mapping.load_state_dict(torch.load('./pretrained_networks/'+model_name+'_mapping.pt'))
g_synthesis.load_state_dict(torch.load('./pretrained_networks/'+model_name+'_synthesis.pt'))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#g_all.eval()
#g_all.to(device)
g_mapping.eval()
g_synthesis.eval()


if 1:
    seed = 500
    torch.manual_seed(seed)
    nb_rows = 2
    nb_cols = 5
    nb_samples = nb_rows * nb_cols
    latents = torch.randn((nb_samples, 512), device=device)
    #noise_in = torch.randn(nb_samples, 1, dim, dim, device=device)
    noise_in_np = [np.random.randn(1,1,4,4),np.random.randn(1,1,4,4),np.random.randn(1,1,8,8),np.random.randn(1,1,8,8),np.random.randn(1,1,16,16),np.random.randn(1,1,16,16),np.random.randn(1,1,32,32),np.random.randn(1,1,32,32),np.random.randn(1,1,64,64),np.random.randn(1,1,64,64),np.random.randn(1,1,128,128),np.random.randn(1,1,128,128),np.random.randn(1,1,256,256),np.random.randn(1,1,256,256),np.random.randn(1,1,512,512),np.random.randn(1,1,512,512)]
    #noise_in_np = [np.zeros((1,1,4,4)),np.zeros((1,1,4,4)),np.zeros((1,1,8,8)),np.zeros((1,1,8,8)),np.zeros((1,1,16,16)),np.random.randn(1,1,16,16),np.random.randn(1,1,32,32),np.random.randn(1,1,32,32),np.random.randn(1,1,64,64),np.random.randn(1,1,64,64),np.random.randn(1,1,128,128),np.random.randn(1,1,128,128),np.random.randn(1,1,256,256),np.random.randn(1,1,256,256),np.random.randn(1,1,512,512),np.zeros((1,1,512,512))]
    noise_in = [torch.from_numpy(item).float().cuda() for item in noise_in_np]
    #print(noise_in.shape); print(noise_in.dtype)
    dlatents = g_mapping(latents).reshape((nb_samples,1,512))
    dlatents = dlatents.expand(-1, num_w_layers, -1)
    print('w shape = '+str(dlatents.shape))

    with torch.no_grad():
        #imgs = g_all(latents,noise_in)
        #imgs = g_all(latents)
        imgs = g_synthesis(dlatents,noise_in)
        #imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
        imgs = (imgs + 1) / 2.0 # normalization to 0..1 range (no clamping)
    imgs = imgs.cpu()

    print(imgs.min())
    imgs_np = imgs.numpy()
    #np.save(f'./paper_figures/fakes_seed_{seed}.npy',imgs_np)
    np.save(f'./fakes_seed_{seed}.npy',imgs_np)
    imgs = torchvision.utils.make_grid(imgs, nrow=nb_cols)

    pyplot.figure(figsize=(15, 6))
    pyplot.imshow(imgs.permute(1, 2, 0).detach().numpy())
    #pyplot.savefig(f'test_generated_images_{seed}.png')

    pyplot.show()

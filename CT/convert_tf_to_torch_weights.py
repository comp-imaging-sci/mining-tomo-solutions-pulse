# Convert trained weights from tf model to torch
# https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

import torch
import dnnlib
import dnnlib.tflib
import pickle
import torch.nn as nn
import collections
from collections import OrderedDict
import stylegan
from stylegan import G_mapping, G_synthesis

model_name = 'NIH_CT_60000'
#model_name = 'MRI_axial_sgan2'

g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis())    
]))

dnnlib.tflib.init_tf()
#weights = pickle.load(open('./pretrained_networks/karras2019stylegan-ffhq-1024x1024.pkl','rb'))
weights = pickle.load(open('./pretrained_networks/'+model_name+'.pkl','rb'))
weights_pt = [collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k,v in w.trainables.items()]) for w in weights]
torch.save(weights_pt, './pretrained_networks/'+model_name+'.pt')

# then on the PyTorch side run
state_G, state_D, state_Gs = torch.load('./pretrained_networks/'+model_name+'.pt')

# def key_translate(k):
#     k = k.lower().split('/')
#     if k[0] == 'g_synthesis':
#         if not k[1].startswith('torgb'):
#             k.insert(1, 'blocks')
#         k = '.'.join(k)
#         k = (k.replace('const.const','const').replace('const.bias','bias').replace('const.stylemod','epi1.style_mod.lin')
#                 .replace('const.noise.weight','epi1.top_epi.noise.weight')
#                 .replace('conv.noise.weight','epi2.top_epi.noise.weight')
#                 .replace('conv.stylemod','epi2.style_mod.lin')
#                 .replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight')
#                 .replace('conv0_up.stylemod','epi1.style_mod.lin')
#                 .replace('conv1.noise.weight', 'epi2.top_epi.noise.weight')
#                 .replace('conv1.stylemod','epi2.style_mod.lin')
#                 .replace('torgb_lod0','torgb'))
#     else:
#         k = '.'.join(k)
#     return k

def key_translate(k):
    k = k.lower().split('/')
    if k[0] == 'g_synthesis':
        if not k[1].startswith('torgb'):
            k.insert(1, 'blocks')
        k = '.'.join(k)
        k = (k.replace('const.const','const').replace('const.bias','bias').replace('const.stylemod','epi1.style_mod.lin')
                .replace('const.noise.weight','epi1.noise.weight')
                .replace('conv.noise.weight','epi2.noise.weight')
                .replace('conv.stylemod','epi2.style_mod.lin')
                .replace('conv0_up.noise.weight', 'epi1.noise.weight')
                .replace('conv0_up.stylemod','epi1.style_mod.lin')
                .replace('conv1.noise.weight', 'epi2.noise.weight')
                .replace('conv1.stylemod','epi2.style_mod.lin')
                .replace('torgb_lod0','torgb'))
    else:
        k = '.'.join(k)
    return k

def weight_translate(k, w):
    k = key_translate(k)
    if k.endswith('.weight'):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w

# we delete the useless torgb filters
param_dict = {key_translate(k) : weight_translate(k, v) for k,v in state_Gs.items() if 'torgb_lod' not in key_translate(k)}
if 1:
    sd_shapes = {k : v.shape for k,v in g_all.state_dict().items()}
    param_shapes = {k : v.shape for k,v in param_dict.items() }

    for k in list(sd_shapes)+list(param_shapes):
        pds = param_shapes.get(k)
        sds = sd_shapes.get(k)
        if pds is None:
            print ("sd only", k, sds)
        elif sds is None:
            print ("pd only", k, pds)
        elif sds != pds:
            print ("mismatch!", k, pds, sds)

g_all.load_state_dict(param_dict, strict=False) # needed for the blur kernels
g_mapping = g_all[0]
g_synthesis = g_all[1]
torch.save(g_all.state_dict(), './pretrained_networks/'+model_name+'.for_g_all.pt')
torch.save(g_mapping.state_dict(), './pretrained_networks/'+model_name+'_mapping.pt')
torch.save(g_synthesis.state_dict(), './pretrained_networks/'+model_name+'_synthesis.pt')
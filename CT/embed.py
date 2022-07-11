from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import time
import torch
import math
import argparse
import os

def ramp_cosine_lr(x,steps,rampdown=0.75,rampup=0.05):
    t = x/steps
    lr_ramp = min(1,(1-t)/rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return lr_ramp

def l2_loss(output,target):
    loss = torch.sum((output - target)**2)
    return loss

parser = argparse.ArgumentParser()
parser.add_argument('-img_dir', type=str, default='real_1', help='Directory of the real image to embed')
parser.add_argument('-model_name', type=str, default='NIH_CT_60000', help='StyleGAN model')
parser.add_argument('-num_restarts', type=int, default=10, help='Number of restarts')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate for Adam')
parser.add_argument('-steps', type=int, default=10000, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='linear1cycledrop, rampcosine,fixed')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 16 times')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
args = parser.parse_args()

schedule_dict = {
    'fixed': lambda x: 1,
    'linear1cycle': lambda x: (9*(1-np.abs(x/args.steps-1/2)*2)+1)/10,
    'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*args.steps)-1/2)*2)+1)/10 if x < 0.9*args.steps else 1/10 + (x-0.9*args.steps)/(0.1*args.steps)*(1/1000-1/10),
    'rampcosine': lambda x: ramp_cosine_lr(x,args.steps)
}
schedule_func = schedule_dict[args.lr_schedule]

# Load reference image
ref_im_np = np.load(args.img_dir+'/gt.npy')
ref_im = torch.FloatTensor(ref_im_np).to('cuda')
ref_im = ref_im.view(1,1,512,512)

# Create directory to save the embedded images
#embed_dir = 'embed_dir_'+args.img_dir+'_2/'
embed_dir = 'embed_dir_'+args.img_dir+'_lr_'+str(args.learning_rate)+'_'+args.lr_schedule+'_2/'
#embed_dir = 'noise_fixed_lr_0.1_tiled_embed_dir_'+args.img_dir+'/'
if not os.path.exists(embed_dir):
    os.mkdir(Path(embed_dir))

synthesis = G_synthesis().cuda()
num_w_layers = 16
print("Loading Synthesis Network")
synthesis.load_state_dict(torch.load('./pretrained_networks/'+args.model_name+'_synthesis.pt'))
for param in synthesis.parameters():
    param.requires_grad = False
lrelu = torch.nn.LeakyReLU(negative_slope=0.2) 
gaussian_fit = torch.load(args.model_name+"_gaussian_fit.pt")

for restart in range(args.num_restarts):
    print(f'\n Starting restart {restart}')

    # Randomly initialize latent tensor
    if args.tile_latent:
        latent = torch.randn(
                (1, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
    else:
        latent = torch.randn((1, num_w_layers, 512), dtype=torch.float, requires_grad=True, device='cuda')

    # Generate list of noise tensors
    noise = [] # stores all of the noise tensors
    noise_vars = []  # stores the noise tensors that we want to optimize on
    noise_type = 'fixed'
    #noise_type = 'trainable'
    #noise_type = args.noise_type

    for i in range(num_w_layers):
        # dimension of the ith noise tensor
        res = (1, 1, 2**(i//2+2), 2**(i//2+2))

        #if(noise_type == 'zero' or i==15):
        if(noise_type == 'zero'):
            new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
            new_noise.requires_grad = False
        elif(noise_type == 'fixed'):
            new_noise = torch.randn(res, dtype=torch.float, device='cuda')
            new_noise.requires_grad = False
        elif (noise_type == 'trainable'):
            new_noise = torch.randn(res, dtype=torch.float, device='cuda')
            if (i < args.num_trainable_noise_layers):
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False
        else:
            raise Exception("unknown noise type")

        noise.append(new_noise)

    var_list = [latent]+noise_vars
    opt = torch.optim.Adam(var_list,lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule_func)

    start_t = time.time()
    print('Starting optimization')
    for j in range(args.steps):
        opt.zero_grad()
        # Duplicate latent in case tile_latent = True
        if (args.tile_latent):
            #latent_in = latent.expand(-1, 18, -1)
            latent_in = latent.expand(-1, num_w_layers, -1)
        else:
            latent_in = latent
        # Apply learned linear mapping to match latent distribution to that of the mapping network
        # latent_in = lrelu(latent_in*gaussian_fit["std"] + gaussian_fit["mean"])
        # Normalize image to [0,1] instead of [-1,1]
        gen_im = (synthesis(latent_in, noise)+1)/2
        loss = l2_loss(gen_im,ref_im)
        if j%100==0:
            print('Iter =  %d, L2 loss = %.2f'%(j,loss.item()))
        loss.backward()
        opt.step()
        scheduler.step()

    total_t = time.time()-start_t
    print(f'\nTotal time taken = {total_t} secs.')

    # Save the embedded image
    embedded_img = gen_im.detach().cpu().numpy()
    embedded_img = np.squeeze(embedded_img)
    np.save(f'{embed_dir}img_{restart}.npy',embedded_img)

print('\nFinished all restarts')










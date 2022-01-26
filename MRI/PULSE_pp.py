from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
from SphericalOptimizer import HollowBallOptimizerDelta
from pathlib import Path
import numpy as np
import time
import torch
from loss_pp import LossBuilder
from functools import partial
import pickle
import math


class PULSE_pp(torch.nn.Module):
    def __init__(self, cache_dir, model_name, num_w_layers, mask_type, verbose=True):
        super(PULSE_pp, self).__init__()

        self.synthesis = G_synthesis().cuda()
        self.verbose = verbose

        self.model_name = model_name
        self.num_w_layers = num_w_layers
        self.mask_type = mask_type

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok = True)
        if self.verbose: print("Loading Synthesis Network")
        assert Path('./'+self.model_name+'_synthesis.pt').exists(),"Synthesis network not in path"
        self.synthesis.load_state_dict(torch.load('./'+self.model_name+'_synthesis.pt'))

        for param in self.synthesis.parameters():
            param.requires_grad = False

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        assert Path(model_name+"_gaussian_fit.pt").exists(),"Fitted Gaussian parameters not in path"
        self.gaussian_fit = torch.load(self.model_name+"_gaussian_fit.pt")

    def forward(self, ref_im,
                seed,
                loss_str,
                eps,
                sigma,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                c,
                steps,
                lr_schedule,
                save_intermediate,
                save_interval,
                **kwargs):

        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]

        # Cut-off values for norms of whitened latent variables
        delta_min_dict = pickle.load(open('./delta_min.pkl','rb'))
        delta_max_dict = pickle.load(open('./delta_max.pkl','rb'))
        delta_min = delta_min_dict[c]
        delta_max = delta_max_dict[c]

        # Generate latent tensor
        if(tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, self.num_w_layers, 512), dtype=torch.float, requires_grad=True, device='cuda')

        # Generate list of noise tensors
        noise = [] # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        for i in range(self.num_w_layers):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))

            if(noise_type == 'zero' or i in [int(layer) for layer in bad_noise_layers.split('.')]):
                new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif(noise_type == 'fixed'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif (noise_type == 'trainable'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                if (i < num_trainable_noise_layers):
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        var_list = [latent]+noise_vars

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'rmsprop': torch.optim.RMSprop,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = HollowBallOptimizerDelta(opt_func, var_list, delta_min=delta_min, delta_max=delta_max, lr=learning_rate)

        def ramp_cosine_lr(x,steps,rampdown=0.75,rampup=0.05):
            t = x/steps
            lr_ramp = min(1,(1-t)/rampdown)
            lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
            lr_ramp = lr_ramp * min(1, t / rampup)
            return lr_ramp
            
        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
            'rampcosine': lambda x: ramp_cosine_lr(x,steps),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)
        
        mask = np.load('./masks/'+self.mask_type+'.npy')
        loss_builder = LossBuilder(ref_im, mask, loss_str, eps, sigma).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        # Arrays for tracking optimization problem
        total_loss_array = np.array([],dtype=np.float32)
        loss_l2_array = np.array([],dtype=np.float32)
        latent_norm_array = np.array([],dtype=np.float32)

        if self.verbose: print("Optimizing")
        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, self.num_w_layers, -1)
            else:
                latent_in = latent

            latent_norm_max = torch.max(torch.norm(latent_in,dim=(0,2)))

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (self.synthesis(latent_in, noise)+1)/2

            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, noise_vars, gen_im, mask)
            loss_dict['TOTAL'] = loss
            total_loss_array = np.append(total_loss_array,loss.cpu().detach())

            # Save best summary for log
            if(loss < min_loss):
                min_loss = loss
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            loss_l2 = loss_dict['L2']
            loss_l2_array = np.append(loss_l2_array,loss_l2.cpu().detach())

            if(loss_l2 < min_l2):
                min_l2 = loss_l2

            if j%save_interval==0:
                print('Iter =  %d, Total loss = %.2f, L2 loss = %.2f'%(j,loss,loss_l2))

            # Save intermediate HR and LR images
            if(save_intermediate):
                yield (gen_im.cpu().detach(),loss_builder.D(gen_im,mask).cpu().detach(),best_im.cpu().detach(),total_loss_array,loss_l2_array)

            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary+current_info)
        yield (best_im.cpu().detach(),total_loss_array,loss_l2_array)
        if(min_l2 > eps):
            print("Could not find an object that downscales correctly within epsilon")


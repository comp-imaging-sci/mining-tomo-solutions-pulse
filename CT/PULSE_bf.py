from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer, SphericalOptimizerNoise, HollowBallOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url


class PULSE_bf(torch.nn.Module):
    #def __init__(self, cache_dir, verbose=True):
    def __init__(self, cache_dir, model_name, num_w_layers, mask_type, verbose=True):
        super(PULSE_bf, self).__init__()

        self.synthesis = G_synthesis().cuda()
        self.verbose = verbose

        #model_name = 'MRI_axial_256x256_norm'
        #num_w_layers = 14
        self.model_name = model_name
        self.num_w_layers = num_w_layers
        self.mask_type = mask_type

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok = True)
        if self.verbose: print("Loading Synthesis Network")
        #with open_url("https://drive.google.com/uc?id=1TCViX1YpQyRsklTVYEJwdbmK91vklCo8", cache_dir=cache_dir, verbose=verbose) as f:
        #    self.synthesis.load_state_dict(torch.load(f))
        self.synthesis.load_state_dict(torch.load('./pretrained_networks/'+self.model_name+'_synthesis.pt'))

        for param in self.synthesis.parameters():
            param.requires_grad = False

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.mapping = G_mapping().cuda()

        #if Path(model_name+"_gaussian_better_fit.pt").exists():
        #    self.gaussian_fit = torch.load(self.model_name+"_gaussian_better_fit.pt")
        #    self.pca = torch.load(self.model_name+"_pca_better_fit.pt")
        #    pass
        if 0:
            pass
        else:
            if self.verbose: print("\tLoading Mapping Network")
            #mapping = G_mapping().cuda()

            # with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
            #         mapping.load_state_dict(torch.load(f))
            self.mapping.load_state_dict(torch.load('./pretrained_networks/'+self.model_name+'_mapping.pt'))

            if self.verbose: print("\tRunning Mapping Network")
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn((1000000,512),dtype=torch.float32, device="cuda")
                latent_out = torch.nn.LeakyReLU(5)(self.mapping(latent))
                #latent_out_centered = latent_out - torch.mean(latent_out,dim=-1).unsqueeze(-1)
                #_,S,V = torch.svd(latent_out_centered)
                #Lambda = S / np.sqrt(latent.shape[0]-1)
                #latent_out_cov = self.cov(latent_out)
                latent_out_cov = torch.load(self.model_name+'_cov.pt')
                egn_val, egn_vec = torch.eig(latent_out_cov,eigenvectors=True)
                egn_val = egn_val.narrow(-1,0,1)
                egn_vec = egn_vec.to('cuda')
                Lambda = torch.sqrt(egn_val).to('cuda')
                #print(latent_out.shape)
                #self.gaussian_fit = {"mean": latentv_out.mean(0), "cov": self.cov(latent_out)}
                self.gaussian_fit = {"mean": latent_out.mean(0).unsqueeze(0)}
                self.pca = {"C":egn_vec,"Lambda":Lambda}
                torch.save(self.gaussian_fit,self.model_name+"_gaussian_better_fit.pt")
                torch.save(self.pca,self.model_name+"_pca_better_fit.pt")
                if self.verbose: print("\tSaved \"gaussian_fit.pt\"")

    def cov(X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1/(D-1) * X @ X.transpose(-1, -2)

    # Function to update latent w when better fitting instead of spherical projection
    #def update_w

    def forward(self, ref_im,
                seed,
                loss_str,
                loss_domain,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                fraction,
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

        # Generate latent tensor
        if(tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, self.num_w_layers, 512), dtype=torch.float, requires_grad=True, device='cuda')

        # latent_z = torch.randn(
        #         (batch_size, 512), dtype=torch.float, requires_grad=False, device='cuda')
        # latent_w = self.mapping(latent_z)
        # latent_v = torch.nn.LeakyReLU(5)(latent_w)
        # print(latent_v.shape); print(self.gaussian_fit["mean"].shape)
        # print(torch.t(self.pca["C"]).shape)
        # print(torch.t(latent_v - self.gaussian_fit["mean"]).shape)
        # print((1./self.pca["Lambda"]).shape)
        # Lambda = self.pca["Lambda"]
        # Lambda_inv = (1./self.pca["Lambda"]).reshape(1,512).expand(batch_size,-1)
        # #latent = torch.matmul(1./self.pca["Lambda"],torch.matmul(torch.t(self.pca["C"]),torch.t(latent_v - self.gaussian_fit["mean"]))) # v^+
        # latent_v_plus = Lambda_inv * torch.t(torch.matmul(torch.t(self.pca["C"]),torch.t(latent_v - self.gaussian_fit["mean"]))) # v^+
        # #latent_v_plus = torch.t(latent_v_plus).reshape(batch_size,1,512)
        # latent_v_plus = latent_v_plus.reshape(batch_size,1,512)
        # if (tile_latent):
        #     latent = latent_v_plus.detach().clone().requires_grad_(True)
        # else:
        #     latent_v_plus = latent_v_plus.expand(-1, self.num_w_layers, -1)
        #     latent = latent_v_plus.detach().clone().requires_grad_(True)

        # if(tile_latent):
        #     latent_z = torch.randn(
        #         (batch_size, 512), dtype=torch.float, requires_grad=False, device='cuda')
        #     latent_w = mapping(latent_z).reshape(batch_size,1,512)
        #     latent_v = self.lrelu(latent_w)
        #     latent = 1./self.pca["Lambda"] * self.gaussian_fit["cov"] * (latent_v - self.gaussian_fit["mean"])
        #     latent = torch.randn(
        #         (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        # else:
        #     latent = torch.randn(
        #         (batch_size, self.num_w_layers, 512), dtype=torch.float, requires_grad=True, device='cuda')

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
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        #opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)
        opt = HollowBallOptimizer(opt_func, var_list, fraction=fraction, lr=learning_rate)
        
        # Distribution parameter
        Lambda = self.pca["Lambda"]

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)
        
        mask = np.load('./masks/'+self.mask_type+'.npy')
        loss_builder = LossBuilder(ref_im, mask, loss_str, loss_domain, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None


        if self.verbose: print("Optimizing")
        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                #latent_in = latent.expand(-1, 18, -1)
                latent_in = latent.expand(-1, self.num_w_layers, -1).clone()
            else:
                latent_in = latent.clone()

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            #latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])
            for layer in range(self.num_w_layers):
                latent_in[:,layer,:] = torch.t(self.lrelu(torch.t(self.gaussian_fit["mean"]) + 
                                            torch.matmul(self.pca["C"],Lambda.reshape(512,1).expand(-1,batch_size)*torch.t(latent_in[:,layer,:]))))

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (self.synthesis(latent_in, noise)+1)/2

            # Calculate Losses
            #loss, loss_dict = loss_builder(latent_in, gen_im)
            loss, loss_dict = loss_builder(latent_in, gen_im, mask)
            loss_dict['TOTAL'] = loss

            # Save best summary for log
            if(loss < min_loss):
                min_loss = loss
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            loss_l2 = loss_dict['L2']

            if(loss_l2 < min_l2):
                min_l2 = loss_l2

            if j%save_interval==0:
                print('Iter =  %5d,L2 loss = %.6f'%(j,loss_l2))

            # Save intermediate HR and LR images
            if(save_intermediate):
                yield (gen_im.cpu().detach().clamp(0, 1),loss_builder.D(gen_im,mask).cpu().detach().clamp(0, 1))

            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary+current_info)
        if(min_l2 <= eps):
            yield (best_im.clone().cpu().detach().clamp(0, 1),loss_builder.D(best_im,mask).cpu().detach().clamp(0, 1))
        else:
            print("Could not find an object that downscales correctly within epsilon")

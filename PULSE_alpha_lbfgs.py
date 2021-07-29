from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
#from SphericalOptimizer import HollowBallOptimizerAlphaLBFGS
from SphericalOptimizer import HollowBallOptimizer2
from pathlib import Path
import numpy as np
import time
import torch
from torch.autograd import Variable
from loss import LossBuilder
from functools import partial
from drive import open_url
import pickle


class PULSE_alpha_lbfgs(torch.nn.Module):
    #def __init__(self, cache_dir, verbose=True):
    def __init__(self, cache_dir, model_name, num_w_layers, mask_type, better_fit=False, verbose=True):
        super(PULSE_alpha_lbfgs, self).__init__()

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

        if Path(model_name+"_gaussian_fit.pt").exists():
            self.gaussian_fit = torch.load(self.model_name+"_gaussian_fit.pt")
        #if 0:
        #    pass
        else:
            if self.verbose: print("\tLoading Mapping Network")
            mapping = G_mapping().cuda()

            # with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
            #         mapping.load_state_dict(torch.load(f))
            mapping.load_state_dict(torch.load('./pretrained_networks/'+self.model_name+'_mapping.pt'))

            if self.verbose: print("\tRunning Mapping Network")
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn((1000000,512),dtype=torch.float32, device="cuda")
                latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
                latent_out_np = latent_out.detach().clone().cpu()
                np.save('latents.npy',latent_out_np)
                self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
                torch.save(self.gaussian_fit,self.model_name+"_gaussian_fit.pt")
                if self.verbose: print("\tSaved \"gaussian_fit.pt\"")
        
    def cov(X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1/(D-1) * X @ X.transpose(-1, -2)

    def jacobian(y, x, create_graph=False):                                                               
        jac = []                                                                                          
        flat_y = y.reshape(-1)                                                                            
        grad_y = torch.zeros_like(flat_y)                                                                 
        for i in range(len(flat_y)):                                                                      
            grad_y[i] = 1.                                                                                
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
        return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
    def hessian(y, x):                                                                                    
        return jacobian(jacobian(y, x, create_graph=True), x)   

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
                p,
                noise_CI_alpha,
                steps,
                lr_schedule,
                better_fit,
                save_intermediate,
                save_interval,
                **kwargs):

        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]

        # Cut-off values for norms of whitened latent variables
        alpha_min_vals = pickle.load(open('./latent_analysis_final/no_bf_alpha_min.pkl','rb'))
        alpha_max_vals = pickle.load(open('./latent_analysis_final/no_bf_alpha_max.pkl','rb'))
        alpha_min = alpha_min_vals[p]
        alpha_max = alpha_max_vals[p]

        # Critical values for noise variables at different resolutions
        noise_CI_min_vals = pickle.load(open('noise_CI_'+str(noise_CI_alpha)+'_min.pkl','rb'))
        noise_CI_max_vals = pickle.load(open('noise_CI_'+str(noise_CI_alpha)+'_max.pkl','rb'))

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
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax,
            'lbfgs' : torch.optim.LBFGS
        }
        opt_func = opt_dict[opt_name]
        #opt = HollowBallOptimizerAlphaLBFGS(opt_func, var_list, alpha_min=alpha_min, alpha_max=alpha_max, lr=learning_rate, history_size=10, line_search_fn='strong_wolfe')
        opt = HollowBallOptimizer2(opt_func, var_list, alpha_min=alpha_min, alpha_max=alpha_max, noise_CI_min_vals=noise_CI_min_vals, noise_CI_max_vals=noise_CI_max_vals, lr=learning_rate, history_size=10, line_search_fn='strong_wolfe')
        
        # schedule_dict = {
        #     'fixed': lambda x: 1,
        #     'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
        #     'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        # }
        # schedule_func = schedule_dict[lr_schedule]
        #scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)
        
        mask = np.load('./masks/'+self.mask_type+'.npy')
        #print(ref_im.shape)
        loss_builder = LossBuilder(ref_im, mask, loss_str, loss_domain, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        # Arrays for tracking optimization problem
        #current_grad_norm_array = np.array([],dtype=np.float32)
        grad_norm_array = np.array([],dtype=np.float32)
        total_loss_array = np.array([],dtype=np.float32)
        loss_l2_array = np.array([],dtype=np.float32)
        w_latent_active = np.array([],dtype=np.int32)
        noise_latent_active = np.array([],dtype=np.int32)

        if self.verbose: print("Optimizing")
        for j in range(steps):

            # Previous iterate variable values
            var_prev = []
            for var in var_list:
                #print('Shape of param = '+str(var_list[idx].shape))
                var_prev.append(var.detach().clone())
                #print(var_prev[idx].shape)

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                #latent_in = latent.expand(-1, 18, -1)
                latent_in = latent.expand(-1, self.num_w_layers, -1)
            else:
                latent_in = latent

            def closure():
                opt.opt.zero_grad()
                if (tile_latent):
                #latent_in = latent.expand(-1, 18, -1)
                    latent_in = latent.expand(-1, self.num_w_layers, -1)
                else:
                    latent_in = latent
                latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])
                gen_im = (self.synthesis(latent_in, noise)+1)/2
                loss, _ = loss_builder(latent_in, gen_im, mask)
                loss.backward()
                return loss
            
            #opt.opt.zero_grad()
            loss = opt.step(closure)

            # Check which constraints are active

            # 1. w latent vector
            latent_mod = (var_prev[0].pow(2).sum(tuple(range(2,var_prev[0].ndim)),keepdim=True)+1e-9).sqrt()
            w_inactive = 0
            for layer in range(self.num_w_layers):
                if (latent_mod[:,layer,:] < alpha_min) or (latent_mod[:,layer,:] > alpha_max):
                    w_latent_active = np.append(w_latent_active,1)
                    print('w latent vector is active')
                    break
                else:
                    w_inactive +=1
            if w_inactive == self.num_w_layers:
                w_latent_active = np.append(w_latent_active,0)
                print('w latent vector is inactive')

            # 2. Noise latent vector
            noise_inactive = 0
            for eta in var_prev[1:]:
                noise_mod = (eta.pow(2)+1e-9).sum().sqrt()
                noise_dim = torch.numel(eta)
                noise_CI_min = noise_CI_min_vals[noise_dim]
                noise_CI_max = noise_CI_max_vals[noise_dim]
                if (noise_mod < noise_CI_min) or (noise_mod > noise_CI_max):
                    noise_latent_active = np.append(noise_latent_active,1)
                    print('noise latent vector is active')
                    break 
                else:
                    noise_inactive +=1
            if noise_inactive == len(var_prev)-1:
                noise_latent_active = np.append(noise_latent_active,0)
                print('noise latent vector is inactive')

            # Current step size
            state = opt.opt.state_dict()['state']
            for key,item in state.items():
                step_size = item['t']
            
            # Projection in the annulus region
            opt.project()
            
            # Compute the norm of the projected gradient
            # Projected gradient = (x-x_prev)/step_size
            grad_norm = 0
            for idx in range(len(var_list)):
                var_curr_idx = var_list[idx].detach().clone()
                var_prev_idx = var_prev[idx]
                grad_norm = grad_norm + torch.sum((var_curr_idx-var_prev_idx)**2)
            grad_norm = (grad_norm ** 0.5)/step_size
            grad_norm_np = grad_norm.cpu().detach().numpy()
            if j==0:
                print('Initial gradient norm = '+str(grad_norm_np))
            else:
                print('Current gradient norm = '+str(grad_norm_np))
                print('\n')
            grad_norm_array = np.append(grad_norm_array,grad_norm_np)
            
            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                #latent_in = latent.expand(-1, 18, -1)
                latent_in = latent.expand(-1, self.num_w_layers, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (self.synthesis(latent_in, noise)+1)/2

            # Calculate Losses
            #loss, loss_dict = loss_builder(latent_in, gen_im)
            #loss, loss_dict = loss_builder(latent_in, gen_im, mask)
            _, loss_dict = loss_builder(latent_in, gen_im, mask)
            #loss, loss_dict = closure()
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
                print('\nIter =  %5d, Total loss = %.7f, L2 loss = %.7f'%(j,loss,loss_l2))

            # Save intermediate HR and LR images
            if(save_intermediate):
                #yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im,mask).cpu().detach().clamp(0, 1))
                #yield (gen_im.cpu().detach(),loss_builder.D(gen_im,mask).cpu().detach())
                yield (gen_im.cpu().detach(),loss_builder.D(gen_im,mask).cpu().detach(),grad_norm_array,total_loss_array,loss_l2_array,w_latent_active,noise_latent_active)
                #yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im,mask).cpu().detach().clamp(0, 1))

            # Termination condition based on the norm of the projected gradient
            if grad_norm_np <= 1e-6 * grad_norm_array[0]:
                print('Solution has converged')
                break

            # if min_l2 <= eps:
            #     print('Solution found within Morozov tolerance')
            #     break

            #loss.backward()
            #opt.step()
            #opt.step(closure)
            #scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary+current_info)
        #yield (best_im.clone().cpu().detach(),loss_builder.D(best_im,mask).cpu().detach())
        if(min_l2 > eps):
            print("Could not find an object that downscales correctly within epsilon")
            

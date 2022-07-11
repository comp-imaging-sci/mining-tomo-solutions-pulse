# Replace downsampling by H^\dagger * H

import torch
from projector_airt import projector
#from bicubic import BicubicDownSample


class LossBuilder(torch.nn.Module):
    #def __init__(self, ref_im, loss_str, eps):
    def __init__(self, ref_im, loss_str, loss_domain, I, n_proj, mu_max, H_csc, eps):
        super(LossBuilder, self).__init__()
        #assert ref_im.shape[2]==ref_im.shape[3]
        assert ref_im.shape[2] == 768*n_proj
        #im_size = ref_im.shape[3]
        self.D = projector(loss_domain=loss_domain,mu_max=mu_max,H_csc=H_csc)
        self.I = I
        self.ref_im = ref_im
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        #self.df_constant = df_constant
        self.eps = eps

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_kl(self, gen_im_lr, ref_im, random_indices, **kwargs):
        random_ref_im = torch.index_select(ref_im,2,random_indices)
        # print(random_ref_im.shape)
        # print(gen_im_lr.shape)
        # print(random_ref_im.max())
        random_gen_im_lr = torch.index_select(gen_im_lr,2,random_indices)
        df_constant_i = torch.where(random_ref_im>0,random_ref_im*torch.log(random_ref_im/self.I)-random_ref_im,torch.zeros(random_ref_im.shape,device='cuda'))
        #df_constant_i = torch.where(random_ref_im>0,random_ref_im*torch.log(random_ref_im/self.I)-random_ref_im,torch.FloatTensor(0).cuda())
        df_constant = torch.sum(df_constant_i)
        #print(df_constant)
        loss_val = torch.sum(random_ref_im*random_gen_im_lr) + self.I * torch.sum(torch.exp(-random_gen_im_lr)) + df_constant
        return loss_val

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        #print((gen_im_lr - ref_im).pow(2).shape)
        #loss_val = ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())
        loss_val = (gen_im_lr - ref_im).pow(2).sum()
        #print(torch.max(gen_im_lr))
        #return ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())
        #print((gen_im_lr - ref_im).pow(2).mean((1,2,3)).sum())
        return loss_val

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10*((gen_im_lr - ref_im).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            #X = latent.view(-1, 1, 18, 512)
            X = latent.view(-1, 1, 16, 512)
            #Y = latent.view(-1, 18, 1, 512)
            Y = latent.view(-1, 16, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D

    # Uses sum of pairwise Euclidean distance between the different extended latent vectors
    def _loss_euclidean(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 16, 512)
            Y = latent.view(-1, 16, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9)
            D = A.sum()
            return D

    # l2 norm on the latent noise variables to be optimized on
    def _loss_noise_l2(self, noise_vars, **kwargs):
        loss_val = 0
        for noise in noise_vars:
            loss_val += 0.5 * (noise.pow(2).sum())
        return loss_val

    #def forward(self, latent, gen_im):
    #def forward(self, latent, gen_im, n_angles):
    def forward(self, latent, noise_vars, gen_im, random_indices):
        var_dict = {'latent': latent,
                    'noise_vars' : noise_vars,
                    'gen_im_lr': self.D(gen_im),
                    'ref_im': self.ref_im,
                    'random_indices': random_indices,
                    }
        loss = 0
        loss_fun_dict = {
            'KL': self._loss_kl,
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
            'EUCLIDEAN': self._loss_euclidean,
            'NOISE_L2': self._loss_noise_l2,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        return loss, losses

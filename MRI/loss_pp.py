# Replace downsampling by H^\dagger * H

import torch
from p_meas import P_meas
from torch.nn import PairwiseDistance

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, mask, loss_str, loss_domain, eps, sigma):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        self.D = P_meas(loss_domain=loss_domain)
        self.ref_im = ref_im
        self.mask = mask
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps
        self.sigma = sigma

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        loss_val = 0.5 * ((gen_im_lr - ref_im).pow(2).sum()) / self.sigma**2
        return loss_val

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10*((gen_im_lr - ref_im).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 14, 512)
            Y = latent.view(-1, 14, 1, 512)
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
            X = latent.view(-1, 1, 14, 512)
            Y = latent.view(-1, 14, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9)
            D = A.sum()
            return D

    # l2 norm on the latent noise variables to be optimized on
    def _loss_noise_l2(self, noise_vars, **kwargs):
        loss_val = 0
        for noise in noise_vars:
            loss_val += 0.5 * (noise.pow(2).sum())
        return loss_val

    
    def forward(self, latent, noise_vars, gen_im, mask):
        var_dict = {'latent': latent,
                    'noise_vars' : noise_vars,
                    'gen_im_lr': self.D(gen_im,self.mask),
                    'ref_im': self.ref_im,
                    }
        loss = 0
        loss_fun_dict = {
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

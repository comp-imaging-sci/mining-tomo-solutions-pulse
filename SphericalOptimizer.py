import math
import torch
from torch.optim import Optimizer

# Spherical Optimizer Class
# Uses the first two dimensions as batch information
# Optimizes over the surface of a sphere using the initial radius throughout
#
# Example Usage:
# opt = SphericalOptimizer(torch.optim.SGD, [x], lr=0.01)

class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}
            # for layer in range(14):
            #     print('Layer = '+str(layer)+', radius = '+str(self.radii[params[0]][:,layer,:]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss

class HollowBallOptimizer(Optimizer):
    def __init__(self, optimizer, params, fraction, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}
            self.eta = {param: self.radii[param]*fraction for param in params}
            for param in params:
                print('Initial shapes:')
                print(param.shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        #print(len(self.params))
        for param in self.params:
            param_mod = (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt()
            # print('Opt shapes')
            # print(param_mod.shape); print(param.shape)
            #if param_mod.shape[1]>1:
            for layer in range(param_mod.shape[1]):
                if param_mod[:,layer,:] <= (self.radii[param][:,layer,:]-self.eta[param][:,layer,:]):
                    param[:,layer,:] = param[:,layer,:] / param_mod[:,layer,:]
                    param[:,layer,:] = param[:,layer,:] * (self.radii[param][:,layer,:]-self.eta[param][:,layer,:])
                elif param_mod[:,layer,:] >= (self.radii[param][:,layer,:]+self.eta[param][:,layer,:]):
                    param[:,layer,:] = param[:,layer,:] / param_mod[:,layer,:]
                    param[:,layer,:] = param[:,layer,:] * (self.radii[param][:,layer,:]+self.eta[param][:,layer,:])
            # if param_mod <= (self.radii[param]-self.eta[param]):
            #     param.data.div_(param_mod); param.mul_(self.radii[param]-self.eta[param])
            # elif param_mod >= (self.radii[param]+self.eta[param]):
            #     param.data.div_(param_mod); param.mul_(self.radii[param]+self.eta[param])
            #param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            #param.mul_(self.radii[param])

        return loss

class HollowBallOptimizerAlpha(Optimizer):
    def __init__(self, optimizer, params, alpha_max, alpha_min, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        with torch.no_grad():
            #self.radii_latent = {param[0]: (param[0].pow(2).sum(tuple(range(2,param[0].ndim)),keepdim=True)+1e-9).sqrt()}
            self.radii_noise = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params[1:]}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        latent_mod = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()
        for layer in range(latent_mod.shape[1]):
            if latent_mod[:,layer,:] <= self.alpha_min:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.alpha_min
            elif latent_mod[:,layer,:] >= self.alpha_max:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.alpha_max
        # Projection of noise variables
        for param in self.params[1:]:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii_noise[param])

        return loss

# Class for hollow ball projection of latent variables and spherical projection of noise variables with L-BFGS optimizer
class HollowBallOptimizerAlphaLBFGS(Optimizer):
    def __init__(self, optimizer, params, alpha_max, alpha_min, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        with torch.no_grad():
            #self.radii_latent = {param[0]: (param[0].pow(2).sum(tuple(range(2,param[0].ndim)),keepdim=True)+1e-9).sqrt()}
            self.radii_noise = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params[1:]}

    def step(self, closure=None):
        loss = self.opt.step(closure)
        return loss

    @torch.no_grad()
    def project(self):
        latent_mod = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()
        for layer in range(latent_mod.shape[1]):
            if latent_mod[:,layer,:] <= self.alpha_min:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.alpha_min
            elif latent_mod[:,layer,:] >= self.alpha_max:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.alpha_max
        # Projection of noise variables
        for param in self.params[1:]:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii_noise[param])

# Class for hollow ball projection of both latent and noise variables with L-BFGS optimizer
# Currently in use
class HollowBallOptimizer2(Optimizer):
    def __init__(self, optimizer, params, alpha_max, alpha_min, noise_CI_min_vals, noise_CI_max_vals, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.noise_CI_min_vals = noise_CI_min_vals
        self.noise_CI_max_vals = noise_CI_max_vals
        # with torch.no_grad():
        #     #self.radii_latent = {param[0]: (param[0].pow(2).sum(tuple(range(2,param[0].ndim)),keepdim=True)+1e-9).sqrt()}
        #     self.radii_noise = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params[1:]}

    def step(self, closure=None):
        loss = self.opt.step(closure)
        return loss

    @torch.no_grad()
    def project(self):
        latent_mod = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()
        for layer in range(latent_mod.shape[1]):
            if latent_mod[:,layer,:] < self.alpha_min:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.alpha_min
            elif latent_mod[:,layer,:] > self.alpha_max:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.alpha_max
        # Projection of noise variables
        for param in self.params[1:]:
            noise_mod = (param.pow(2)+1e-9).sum().sqrt()
            noise_dim = torch.numel(param)
            noise_CI_min = self.noise_CI_min_vals[noise_dim]
            noise_CI_max = self.noise_CI_max_vals[noise_dim]
            if noise_mod < noise_CI_min:
                param = param / noise_mod
                param = param * noise_CI_min
            if noise_mod > noise_CI_max:
                param = param / noise_mod
                param = param * noise_CI_max
                
# Class for spherical projection of noise variables only
class SphericalOptimizerNoise(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params[1:]}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params[1:]:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss
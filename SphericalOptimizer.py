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

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss

# Optimizer class for performing projected gradient descent with spherical constraint only on the intermediate latent vectors
class SphericalOptimizerStyle(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.radii = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        latent_mod = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()
        for layer in range(latent_mod.shape[1]):
            self.params[0][:,layer,:] = self.radii[:,layer,:]*(self.params[0][:,layer,:] / latent_mod[:,layer,:])
        return loss

# Optimizer class for performing projected gradient descent with hollow-ball constraint only on the intermediate latent vectors
class HollowBallOptimizerDelta(Optimizer):
    def __init__(self, optimizer, params, delta_max, delta_min, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.delta_min = delta_min
        self.delta_max = delta_max

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        latent_mod = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()
        for layer in range(latent_mod.shape[1]):
            if latent_mod[:,layer,:] <= self.delta_min:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.delta_min
            elif latent_mod[:,layer,:] >= self.delta_max:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.delta_max

        return loss

# Optimizer class for performing projected gradient descent with hollow-ball constraint on the intermediate latent vectors
# and spherical constraint on the latent noise vectors
class HollowBallOptimizerDelta2(Optimizer):
    def __init__(self, optimizer, params, delta_max, delta_min, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.delta_min = delta_min
        self.delta_max = delta_max
        with torch.no_grad():
            self.radii_noise = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params[1:]}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        # Hollow ball projection of style vectors
        latent_mod = (self.params[0].pow(2).sum(tuple(range(2,self.params[0].ndim)),keepdim=True)+1e-9).sqrt()
        for layer in range(latent_mod.shape[1]):
            if latent_mod[:,layer,:] <= self.delta_min:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.delta_min
            elif latent_mod[:,layer,:] >= self.delta_max:
                self.params[0][:,layer,:] = self.params[0][:,layer,:] / latent_mod[:,layer,:]
                self.params[0][:,layer,:] = self.params[0][:,layer,:] * self.delta_max
        # Spherical projection of noise latent vectors
        for param in self.params[1:]:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii_noise[param])

        return loss

# Test the Chambolle-Pock primal dual algorithm for computing the pseudoinverse solution
# https://blog.allardhendriksen.nl/cwi-ci-group/chambolle_pock_using_tomosipo/
import numpy as np
import torch
from torch_radon import RadonFanbeam

def normalize(x):
    size = x.size()
    x = x.view(size[0], -1)
    norm = torch.norm(x, dim=1)
    x /= norm.view(-1, 1)
    return x.view(*size), torch.max(norm).item()

def operator_norm(img_size,device,n_iter=100,batch_size=1):
    with torch.no_grad():
        x = torch.randn((batch_size, img_size, img_size), device=device)
        #x, _ = normalize(x)
        for i in range(n_iter):
            next_x = self.operator.backward(self.operator.forward(x))
            x, v = normalize(next_x)

        #return 2.0 / v
        next_x = self.operator.backward(self.operator.forward(x))
        return torch.norm(next_x)/


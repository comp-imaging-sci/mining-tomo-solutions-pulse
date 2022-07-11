# Function to get the MNLS solution using Landweber iterations

import torch
from torch_radon import RadonFanbeam
from torch_radon.solvers import Landweber

def mnls(radon_fanbeam,g,device,noiseless=True,delta=None):
    landweber = Landweber(radon_fanbeam,delta=delta,noiseless=noiseless)
    alpha = 0.95 * landweber.estimate_alpha(512, device)
    starting_point = torch.zeros((512,512),dtype=torch.float32).to(device)
    reconstruction, _ = landweber.run(starting_point, g, alpha, iterations=1000, callback=lambda xx: (torch.norm(radon_fanbeam.forward(xx)-g)**2).item())
    f_hat = reconstruction.detach().cpu().numpy()
    return f_hat
# Fanbeam projector class
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_radon import RadonFanbeam

class projector(nn.Module):

    def __init__(self, loss_domain="meas", n_angles = 25, cuda=True):
        super().__init__()
        self.loss_domain = loss_domain
        self.cuda = '.cuda' if cuda else ''
        fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        self.radon_fanbeam = RadonFanbeam(512,fanbeam_angles,source_distance=512, det_distance=512, det_spacing=3.0)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_squeezed = torch.squeeze(x)
        g = self.radon_fanbeam.forward(x_squeezed)
        g = g.unsqueeze(0).unsqueeze(0)
        return g


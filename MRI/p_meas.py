# Class for forward projection
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class P_meas(nn.Module):

    def torch_real_to_complex(self,x):
        x_real = x.to('cuda')
        x_imag = torch.zeros(x.shape,dtype=x.dtype).to('cuda')
        x_cplx = torch.stack((x_real,x_imag),axis=-1)
        return x_cplx

    def H(self, data, mask):
        mask = np.fft.ifftshift(mask)
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.view(1,1,mask.shape[0],mask.shape[1],1).repeat(1,1,1,1,2).to('cuda')
        assert data.size(-1) == 2
        data = torch.fft(data, 2, normalized=True)
        return mask_tensor * data

    def __init__(self, cuda=True):
        super().__init__()
        self.cuda = '.cuda' if cuda else ''
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        f = self.torch_real_to_complex(x)
        g = self.H(f, mask)
        return g


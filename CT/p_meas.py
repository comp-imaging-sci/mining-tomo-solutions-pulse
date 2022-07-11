# Class for projecting onto the measurable component
# H: undersampled 2D FFT
# H_dagger: 2D IFFT
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
        #data = ifftshift(data, dim=(-3, -2))
        data = torch.fft(data, 2, normalized=True)
        #data = fftshift(data, dim=(-3, -2))
        #print(mask.shape)
        #print(data.shape)
        #return torch.where(mask_tensor == 0, torch.Tensor([0]), data)
        return mask_tensor * data

    def H_dagger(self, data):
        assert data.size(-1) == 2
        #data = ifftshift(data, dim=(-3, -2))
        data = torch.ifft(data, 2, normalized=True)
        #data = fftshift(data, dim=(-3, -2))
        return data


    #def __init__(self, factor=4, cuda=True, padding='reflect'):
    def __init__(self, loss_domain="image", cuda=True):
        super().__init__()
        # self.factor = factor
        # size = factor * 4
        # k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
        #                   for i in range(size)], dtype=torch.float32)
        # k = k / torch.sum(k)
        # # k = torch.einsum('i,j->ij', (k, k))
        # k1 = torch.reshape(k, shape=(1, 1, size, 1))
        # self.k1 = torch.cat([k1, k1, k1], dim=0)
        # k2 = torch.reshape(k, shape=(1, 1, 1, size))
        # self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.loss_domain = loss_domain
        self.cuda = '.cuda' if cuda else ''
        #self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    #def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
    def forward(self, x, mask):
        # x = torch.from_numpy(x).type('torch.FloatTensor')
        # filter_height = self.factor * 4
        # filter_width = self.factor * 4
        # stride = self.factor

        # pad_along_height = max(filter_height - stride, 0)
        # pad_along_width = max(filter_width - stride, 0)
        # filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
        # filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))

        # # compute actual padding values for each side
        # pad_top = pad_along_height // 2
        # pad_bottom = pad_along_height - pad_top
        # pad_left = pad_along_width // 2
        # pad_right = pad_along_width - pad_left

        # # apply mirror padding
        # if nhwc:
        #     x = torch.transpose(torch.transpose(
        #         x, 2, 3), 1, 2)   # NHWC to NCHW

        # # downscaling performed by 1-d convolution
        # x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        # x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
        # if clip_round:
        #     x = torch.clamp(torch.round(x), 0.0, 255.)

        # x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        # x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        # if clip_round:
        #     x = torch.clamp(torch.round(x), 0.0, 255.)

        # if nhwc:
        #     x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        # if byte_output:
        #     return x.type('torch.ByteTensor'.format(self.cuda))
        # else:
        #     return x
        f = self.torch_real_to_complex(x)
        g = self.H(f, mask)
        if self.loss_domain == "image":
            f_meas = self.H_dagger(g)
            f_meas = torch.squeeze(f_meas.narrow(-1,0,1),-1)
        else:
            return g


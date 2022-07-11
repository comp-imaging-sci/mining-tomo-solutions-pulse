# Fanbeam projector class
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import scipy.io

def to_sparse_tensor(H_csc):
    H_coo = H_csc.tocoo()
    values = H_coo.data
    indices = np.vstack((H_coo.row,H_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = H_coo.shape
    H = torch.sparse_coo_tensor(i,v,shape)
    return H

class projector(nn.Module):

    def __init__(self, mu_max, H_csc, loss_domain="meas", cuda=True):
        super().__init__()
        self.loss_domain = loss_domain
        self.cuda = 'cuda' if cuda else ''
        #H_csc = scipy.io.loadmat('H_views_'+str(n_angles)+'.mat')['H']
        self.mu_max = mu_max
        H = to_sparse_tensor(H_csc)
        self.H = H.to(self.cuda)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_vec = x.view(512**2,-1)
        g = 0.82*self.mu_max*torch.sparse.mm(self.H,x_vec)
        g = g.unsqueeze(0).unsqueeze(0)
        return g


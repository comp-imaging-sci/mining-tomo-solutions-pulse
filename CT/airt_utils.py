import numpy as np 
import torch 

# Convert sparse scipy system matrix to sparse torch tensor
def to_sparse_tensor(H_csc):
    H_coo = H_csc.tocoo()
    values = H_coo.data
    indices = np.vstack((H_coo.row,H_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = H_coo.shape
    H = torch.sparse_coo_tensor(i,v,shape)
    return H


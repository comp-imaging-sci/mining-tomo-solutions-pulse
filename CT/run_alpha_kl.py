# Modified from run.py 
# Hollow ball optimizer modified based on empirical distribution of whitened latent vectors
# GeoCROSS loss replaced by Euclidean distance

#from PULSE_alpha import PULSE_alpha
from PULSE_alpha_kl import PULSE_alpha_kl
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torch
import torchvision
from math import log10, ceil
import argparse
import numpy as np

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        #self.image_list = list(self.root_path.glob("*.png"))
        self.image_list = list(self.root_path.glob("*.npy"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        #image = torchvision.transforms.ToTensor()(Image.open(img_path))
        image_np = np.load(img_path)
        image_np = image_np[np.newaxis,:,:] # Convert to CxHxW
        #print(image_np.shape)
        #image = torch.from_numpy(image_np)
        image = to_tensor(image_np)
        if(self.duplicates == 1):
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"

parser = argparse.ArgumentParser(description='PULSE')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
parser.add_argument('-output_dir', type=str, default='runs', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')

#PULSE arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
parser.add_argument('-loss_domain', type=str, choices=["image","meas"], default="image", help='Loss function domain')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
#parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-bad_noise_layers', type=str, default="15", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-p', type=float, default=0.1, help='Cumulative probability to decide cut-off for norms of whitened variables')
#parser.add_argument('-noise_CI_alpha', type=float, default=0.95, help='Significance level for confidence interval of noise variables')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')
parser.add_argument('-better_fit', action='store_true', help='Fit more accurate multivariate Gaussian distribution')
parser.add_argument('-save_interval', type=int, default=100, help='Interval for saving intermediate results')

# SB
parser.add_argument('-model_name', type=str, required=True, default='MRI_axial_256x256_norm', help='Name of the model used for the StyleGAN')
parser.add_argument('-num_w_layers', type=int, required=True, default=14, help='Number of w layers')
parser.add_argument('-I', type=float, required=True, help='Incident X-ray intensity')
parser.add_argument('-n_angles', required=True, type=int, default=60, help='Number of views in the fanbeam projector')

kwargs = vars(parser.parse_args())

dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])
#dataloader = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])

#model = PULSE(cache_dir=kwargs["cache_dir"])
if not kwargs["better_fit"]:
    #model = PULSE_alpha(cache_dir=kwargs["cache_dir"],model_name=kwargs["model_name"],num_w_layers=kwargs["num_w_layers"],n_angles=kwargs["n_angles"])
    model = PULSE_alpha_kl(cache_dir=kwargs["cache_dir"],model_name=kwargs["model_name"],num_w_layers=kwargs["num_w_layers"],I=kwargs["I"],n_angles=kwargs["n_angles"])
else:
    model = PULSE_bf_alpha(cache_dir=kwargs["cache_dir"],model_name=kwargs["model_name"],num_w_layers=kwargs["num_w_layers"],n_angles=kwargs["n_angles"])
model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()

alt_count = 0
for ref_im, ref_im_name in dataloader:
    print('\n---------Starting alternate solution '+str(alt_count)+'-----------')
    if(kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        for i in range(kwargs["batch_size"]):
        #for i in range(1):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        #for j,(HR,LR) in enumerate(model(ref_im,**kwargs)):
        #for j,(HR,LR,current_grad_norm,total_loss,loss_l2,w_latent_active,noise_latent_active) in enumerate(model(ref_im,**kwargs)):
        for j,(HR,LR,best_HR,total_loss,loss_kl) in enumerate(model(ref_im,**kwargs)):
            for i in range(kwargs["batch_size"]):
                np.save(out_path / ref_im_name[i] / "best_HR.npy",best_HR)
                np.save(out_path / ref_im_name[i] / "total_loss.npy",total_loss)
                np.save(out_path / ref_im_name[i] / "loss_kl.npy",loss_kl)
                if j%kwargs["save_interval"]==0:
                    np.save(int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.npy",HR[i].cpu().detach())
                    np.save(int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.npy",LR[i].cpu().detach())
                # toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                #     int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                # toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                #     int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
    # else:
    #     #out_im = model(ref_im,**kwargs)
    #     for j,(HR,LR) in enumerate(model(ref_im,**kwargs)):
    #         for i in range(kwargs["batch_size"]):
    #         #for i in range(1):
    #             # toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
    #             #     out_path / f"{ref_im_name[i]}.png")
    else:
        #for j,(HR,LR) in enumerate(model(ref_im,**kwargs)):
        for j,(best_HR,total_loss,loss_kl) in enumerate(model(ref_im,**kwargs)):
            for i in range(kwargs["batch_size"]):
                int_path = Path(out_path / ref_im_name[i])
                int_path.mkdir(parents=True, exist_ok=True)
                np.save(int_path / "best_HR.npy",best_HR)
                np.save(int_path / "total_loss.npy",total_loss)
                np.save(int_path / "loss_kl.npy",loss_kl)
                #np.save(out_path / f"{ref_im_name[i]}.npy",HR[i].cpu().detach())
            #np.save(out_path / f"{ref_im_name[i]}.npy",LR[i].cpu().detach())
    
    alt_count = alt_count + 1

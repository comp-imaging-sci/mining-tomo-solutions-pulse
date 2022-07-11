# Mining the manifolds of deep generative models for multiple data-consistent solutions of tomographic imaging problems - PyTorch implementation

### Paper: https://arxiv.org/abs/2202.05311

Sayantan Bhadra<sup>1</sup>, Umberto Villa<sup>1</sup> and Mark A. Anastasio<sup>2</sup> <br />
<sup>1</sup>Washington University in St. Louis, USA <br />
<sup>2</sup>University of Illinois at Urbana-Champaign, USA

---

![Transformation Preview](https://github.com/comp-imaging-sci/mining-tomo-solutions-pulse/blob/main/figures/mri_panel_8x.png)

## System requirements
* Linux
* Anaconda >= 2018.2 
* Python >= 3.6
* Numpy 1.18.2
* PyTorch 1.3.1
* 1 NVIDIA GPU (compute capability GeForce GTX 1080 or higher and minimum 8 GB RAM)
* NVIDIA driver >= 440.59, CUDA toolkit >= 10.0

## Directory structure and usage
* `MRI`: Contains data and code for undersampled k-space data
* `CT`: Contains data and code for limited-angle CT data

## MRI experiments
* `knee_1` and `knee_2`: Folders containing objects Knee 1 and Knee 2
* `kspace_1` and `kspace_2`: Folders containing undersampled k-space data from Knee 1 and Knee 2
* `masks`: Folder containing 6-fold and 8-fold random Cartesian sampling masks

## CT experiments




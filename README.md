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
* NVIDIA GPU(s) (compute capability GeForce GTX 1080 or higher and minimum 8 GB RAM)
* NVIDIA driver >= 440.59, CUDA toolkit >= 10.0

## Environments
* **PULSE++ and PULSE**: Create a virtual environment with the following command:
```
conda create -f pulse_pp.yml
```
* **Langevin sampling**: Set up virtual environment by running the commands at https://github.com/utcsilab/csgm-mri-langevin#setup-environment, with root directory as `MRI/langevin/`

## Undersampled k-space experiments
The scripts for PULSE++ and PULSE should be run with root directory as `MRI/`. For running Langevin sampling, the root directory should be `MRI/langevin`.

### Data subdirectories in `MRI`:
* `knee_1` and `knee_2`: Folders containing objects Knee 1 and Knee 2
* `kspace_1` and `kspace_2`: Folders containing undersampled k-space data from Knee 1 and Knee 2
* `masks`: Folder containing 6-fold and 8-fold random Cartesian sampling masks

### PULSE++
1. Download `MRI_synthesis.pkl` with MRI-StyleGAN synthesis network model weights from Databank. Save the file under `MRI/` (used for both PULSE++ and PULSE)
2. Run `./run_meas_alpha.sh`

### PULSE
1. Run `./run_meas.sh`

### Langevin sampling
1. Go to the directory `MRI/langevin`
2. Download `MRI_checkpoint.pt` for NCSNv2 model weights from Databank. Save the file in `MRI/langevin`
3. Run `./run_multiple_no_norm.sh`


## Limited-angle CT experiments
Run PULSE++ with root directory as `CT/`

### Data subdirectories in `MRI`:
* `lung_1` and `lung_2`: Folders containing objects Lung 1 and Lung 2
* `proj_1` and `proj_2`: Folders containing limited-angle projection data from Lung 1 and Lung 2

### PULSE++
1. Download the system matrix `H` from Databank
2. Run `run_meas_alpha_airt_kl.sh`

Finish!


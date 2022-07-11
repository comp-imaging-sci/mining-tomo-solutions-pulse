#!/bin/bash
IDX=1
I=1e3
N_ANGLES=25

# Noiseless sinogram
#python align_meas.py -seed 0 -input_dir real_$IDX -output_dir input_g_$IDX -n_angles $N_ANGLES -loss_domain meas

# Noisy sinogram
python align_meas.py -seed 0 -I $I -input_dir real_$IDX -output_dir input_g_${IDX}_I_$I -n_angles $N_ANGLES -loss_domain meas 

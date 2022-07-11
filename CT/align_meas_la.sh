#!/bin/bash
IDX=1
I=1e4
ANGLE_FACTOR=6

# Noiseless sinogram
#python align_meas.py -seed 0 -input_dir real_$IDX -output_dir input_g_$IDX -n_angles $N_ANGLES -loss_domain meas

# Noisy sinogram
python align_meas_la.py -seed 0 -I $I -input_dir real_$IDX -output_dir input_g_${IDX}_I_$I -angle_factor $ANGLE_FACTOR -loss_domain meas 

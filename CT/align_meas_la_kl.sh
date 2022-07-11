#!/bin/bash
IDX=1
I=1e4
ANGLE_FACTOR=6
#python align_meas.py -input_dir real_0 -output_dir input_y_0 -n_angles 25 -loss_domain meas 
python align_meas_la_kl.py -seed 0 -I $I -input_dir real_$IDX -output_dir input_y_${IDX}_I_$I -angle_factor $ANGLE_FACTOR -loss_domain meas 

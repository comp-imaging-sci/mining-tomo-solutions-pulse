#!/bin/bash
#python align_meas.py -input_dir real_g_axial_60351 -output_dir input_g_axial_60351 -mask_type mask_LP_32x32 -loss_domain meas 
#python align_meas.py -input_dir real_g_axial_60010 -output_dir input_g_axial_60010 \
#python align_meas.py -input_dir real_g_axial_60010 -output_dir input_g_axial_60010 -mask_type mask_variable_density_20_fold -loss_domain meas
#python align_meas.py -input_dir real_g_axial_60010 -output_dir input_g_axial_60010 -mask_type mask_random_8_fold_cartesian -loss_domain meas
SNR=10
python align_meas.py -input_dir real_g_axial_60010 -output_dir noisy_snr_${SNR}_input_g_axial_60010 -mask_type mask_random_6_fold_cartesian -loss_domain meas -SNR $SNR
#python align_meas.py -input_dir real_g_axial_60010 -output_dir input_g_axial_60010 -mask_type mask_random_4_fold_cartesian -loss_domain meas 

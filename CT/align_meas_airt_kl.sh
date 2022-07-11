#!/bin/bash
#IDX=1
IDX=3
I=1e5
#N_ANGLES=24
LA=120
#MU_MAX=0.063 # real_1
MU_MAX=0.046 # real_3

# Limited views
#python align_meas_airt_kl.py -seed 0 -I $I -input_dir real_$IDX -output_dir input_airt_y_${IDX}_I_$I -n_angles $N_ANGLES -loss_domain meas
#python align_meas_airt_kl.py -seed 0 -input_dir real_$IDX -output_dir input_airt_y_${IDX} -n_angles $N_ANGLES -loss_domain meas 

# Limited angle
# Real
python align_meas_airt_kl.py -seed 0 -I $I -la $LA -mu_max $MU_MAX -input_dir real_$IDX -output_dir input_airt_y_${IDX}_I_$I -loss_domain meas -limited_angle

# Fake
#python align_meas_airt_kl.py -seed 0 -I $I -la $LA -mu_max $MU_MAX -input_dir fake_0 -output_dir input_airt_y_fake_0_I_$I -loss_domain meas -limited_angle

# Embedded image
#python align_meas_airt_kl.py -seed 0 -I $I -la $LA -mu_max $MU_MAX -input_dir embed_1_2 -output_dir input_airt_y_embed_1_2_I_$I -loss_domain meas -limited_angle
#python align_meas_airt_kl.py -seed 0 -I $I -la $LA -mu_max $MU_MAX -input_dir embed_3_0 -output_dir input_airt_y_embed_1_2_I_$I -loss_domain meas -limited_angle

# Debug (limited angle)
#python align_meas_airt_kl.py -debug -seed 0 -I $I -la $LA -mu_max 0.0122 -input_dir real_$IDX -output_dir input_airt_y_${IDX}_I_$I -loss_domain meas -limited_angle


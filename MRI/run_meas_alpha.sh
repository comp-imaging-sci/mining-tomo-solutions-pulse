#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# I/O arguments
#MASK_TYPE=mask_LP_32x32
#MASK_TYPE=mask_equispaced_8_fold_cartesian_256x256
MASK_TYPE=mask_random_8_fold_cartesian
#MASK_TYPE=mask_random_4_fold_cartesian
#MASK_TYPE=mask_variable_density_20_fold
EPS=3200.0 # SNR = 10, 8x
#EPS=4096.0 # SNR = 10, 6x
#EPS=5248.0 # SNR = 10, 7x
#EPS=43806.0 # SNR = 10, 6x      
LEARNING_RATE=0.4
P=0.001
LAMBDA=1e5 # Euclidean loss hyperparameter
LOSS_STR=1*L2+${LAMBDA}*EUCLIDEAN+1*NOISE_L2
echo $LOSS_STR
#INPUT_DIR=input_g_axial_60010/$MASK_TYPE
SNR=10
#SIGMA=0.07 # Measurement noise standard deviation
SIGMA=0.06999067596309323 # 10 dB
#SIGMA=0.03935864947205416 # 15 dB
#SIGMA=0.05248560261198792 # 12.5 dB
INPUT_DIR=noisy_snr_${SNR}_input_g_axial_60010/$MASK_TYPE
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_alpha/${MASK_TYPE}/lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_alpha/${MASK_TYPE}/no_bf_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/radon/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_alpha_5/${MASK_TYPE}/lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
OUTPUT_DIR=/scratch/pulse_paper/

#OUTPUT_DIR=/scratch/test_bf_alpha
#OUTPUT_DIR=/shared/radon/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_alpha_3/${MASK_TYPE}/lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/curie/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_alpha_2/${MASK_TYPE}/lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR/${MASK_TYPE}/test_bf_alpha
DUPLICATES=100

# PULSE arguments
#EPS=2e-3
#STEPS=10000
STEPS=10000
SAVE_INTERVAL=100

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
MODEL_NAME='MRI_axial_256x256_norm_60000'
NUM_W_LAYERS=14

# Save intermediate
# python -u run_alpha.py -sigma $SIGMA -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
# -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -mask_type $MASK_TYPE -num_w_layers $NUM_W_LAYERS -mask_type $MASK_TYPE -save_intermediate -save_interval $SAVE_INTERVAL

# Save only final
python -u run_alpha.py -sigma $SIGMA -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
-learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -mask_type $MASK_TYPE -num_w_layers $NUM_W_LAYERS -mask_type $MASK_TYPE -save_interval $SAVE_INTERVAL
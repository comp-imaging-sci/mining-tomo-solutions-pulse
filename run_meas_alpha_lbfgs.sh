#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# I/O arguments
#MASK_TYPE=mask_LP_32x32
#MASK_TYPE=mask_equispaced_8_fold_cartesian_256x256
#MASK_TYPE=mask_random_8_fold_cartesian
MASK_TYPE=mask_random_8_fold_cartesian
#MASK_TYPE=mask_random_4_fold_cartesian
#MASK_TYPE=mask_variable_density_20_fold
EPS=0.000611 # SNR = 10, 8x
#EPS=0.0001 # SNR = 10, 8x
#EPS=0.000815 # SNR = 10, 6x
#EPS=0.001223 # SNR = 10, 4x
BATCH_SIZE=1
LEARNING_RATE=0.1
P=0.01
NOISE_CI_ALPHA=0.9
LAMBDA=1e-8 # Euclidean loss hyperparameter
#LOSS_STR=100*L2+${LAMBDA}*EUCLIDEAN
LOSS_STR=1*L2+${LAMBDA}*EUCLIDEAN
echo $LOSS_STR
#INPUT_DIR=input_g_axial_60010/$MASK_TYPE
SNR=10
INPUT_DIR=noisy_snr_${SNR}_input_g_axial_60010/$MASK_TYPE
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_alpha/${MASK_TYPE}/lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_2/${MASK_TYPE}/lbfgs_p_${P}_noise_CI_${NOISE_CI_ALPHA}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR/${MASK_TYPE}/test_noise_CI
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_2/${MASK_TYPE}/test_optim_proj_grad
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_2/${MASK_TYPE}/test_optim_save_active
OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR_2/${MASK_TYPE}/test_optim_save_active_2

DUPLICATES=100

# PULSE arguments
#EPS=2e-3
#STEPS=10000
STEPS=5000
SAVE_INTERVAL=1

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
MODEL_NAME='MRI_axial_256x256_norm_60000'
NUM_W_LAYERS=14

python -u run_alpha.py -batch_size $BATCH_SIZE -opt_name lbfgs -p $P -noise_CI_alpha $NOISE_CI_ALPHA -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
-learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -mask_type $MASK_TYPE -num_w_layers $NUM_W_LAYERS -mask_type $MASK_TYPE -save_interval $SAVE_INTERVAL -save_intermediate 

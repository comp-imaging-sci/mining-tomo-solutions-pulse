#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# I/O arguments
#MASK_TYPE=mask_LP_32x32
#MASK_TYPE=mask_equispaced_8_fold_cartesian_256x256
MASK_TYPE=mask_random_8_fold_cartesian
#MASK_TYPE=mask_random_4_fold_cartesian
#MASK_TYPE=mask_variable_density_20_fold
#EPS=1e-4
LEARNING_RATE=0.4
LAMBDA=1e-3 # GEOCROSS loss hyperparameter
LOSS_STR=100*L2+${LAMBDA}*GEOCROSS
echo $LOSS_STR
#INPUT_DIR=input_g_axial_60010/$MASK_TYPE
SNR=10
INPUT_DIR=noisy_snr_${SNR}_input_g_axial_60010/$MASK_TYPE
#INPUT_DIR=input_g_axial_60546/$MASK_TYPE
#INPUT_DIR=input_g_axial_60351/$MASK_TYPE
#OUTPUT_DIR=runs/meas_AR_${MASK_TYPE}_${EPS}_${LEARNING_RATE}_geocross_$LAMBDA
#OUTPUT_DIR=/shared/curie/MRI/sbhadra/pulse_null_functions/runs/meas_AR/${MASK_TYPE}/lr_${LEARNING_RATE}_geocross_$LAMBDA
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR/${MASK_TYPE}/lr_${LEARNING_RATE}_geocross_$LAMBDA
OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR/${MASK_TYPE}/test
DUPLICATES=1

# PULSE arguments
#EPS=2e-3
#STEPS=10000
STEPS=10
SAVE_INTERVAL=1

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
MODEL_NAME='MRI_axial_256x256_norm_60000'
NUM_W_LAYERS=14

python -u run.py -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps 1 \
-learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -mask_type $MASK_TYPE -num_w_layers $NUM_W_LAYERS -mask_type $MASK_TYPE -save_intermediate -save_interval $SAVE_INTERVAL

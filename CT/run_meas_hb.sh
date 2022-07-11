#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# I/O arguments
N_ANGLES=25
LEARNING_RATE=0.4
FRACTION=0.5
LAMBDA=1e3 # GEOCROSS loss hyperparameter
LOSS_STR=1*L2+${LAMBDA}*GEOCROSS
echo $LOSS_STR
BAD_NOISE_LAYERS=1000
#INPUT_DIR=input_g_axial_60010/$MASK_TYPE
#SNR=10
INPUT_DIR=input_g_0/${N_ANGLES}_views
OUTPUT_DIR=/shared/radon/MRI/sbhadra/pulse_ct/runs/${N_ANGLES}_views/bn_lr_${LEARNING_RATE}_frac_${FRACTION}_geocross_${LAMBDA}
#OUTPUT_DIR=/shared/radon/MRI/sbhadra/pulse_ct/runs/${N_ANGLES}_views/test
DUPLICATES=100

# PULSE arguments
#EPS=2e-3
#STEPS=10000
STEPS=2000
SAVE_INTERVAL=100

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
MODEL_NAME='NIH_CT_5000'
NUM_W_LAYERS=16

python -u run.py -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -bad_noise_layers $BAD_NOISE_LAYERS -eps 1000 -fraction $FRACTION \
-learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -n_angles $N_ANGLES -num_w_layers $NUM_W_LAYERS -save_intermediate -save_interval $SAVE_INTERVAL

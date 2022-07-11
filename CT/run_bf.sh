#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# I/O arguments
MASK_TYPE=mask_LP_32x32
EPS=1e-4
LEARNING_RATE=0.1
LAMBDA=1.0 # GEOCROSS loss hyperparameter
LOSS_STR=100*L2+${LAMBDA}*GEOCROSS
echo $LOSS_STR
INPUT_DIR=input/$MASK_TYPE
OUTPUT_DIR=runs/bf_${MASK_TYPE}_${EPS}_${LEARNING_RATE}_geocross_$LAMBDA
DUPLICATES=1

# PULSE arguments
#EPS=2e-3
STEPS=1000
SAVE_INTERVAL=100

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
MODEL_NAME='MRI_axial_256x256_norm_60000'
NUM_W_LAYERS=14

python -u run.py -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -eps $EPS \
                 -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -mask_type $MASK_TYPE -num_w_layers $NUM_W_LAYERS \
                 -mask_type $MASK_TYPE -save_intermediate -save_interval $SAVE_INTERVAL -better_fit

#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0
MASK_TYPE=mask_random_6_fold_cartesian
#MASK_TYPE=mask_random_8_fold_cartesian
if [ $MASK_TYPE = "mask_random_6_fold_cartesian" ]
then
    EPS=4096 # Acceptance tolerance
else
    EPS=3200
fi   
LEARNING_RATE=0.4
LAMBDA=0.01 # GEOCROSS loss hyperparameter (used in paper for 6-fold mask)
#LAMBDA=0.1 # used in paper for 8-fold mask
LOSS_STR=1*L2+${LAMBDA}*GEOCROSS
echo $LOSS_STR
SIGMA=0.07 # Measurement noise standard deviation
#SIGMA=0.05 
INPUT_DIR=input_g_sigma_${SIGMA}/$MASK_TYPE
OUTPUT_DIR=runs/${MASK_TYPE}/lr_${LEARNING_RATE}_reg_${LAMBDA}
DUPLICATES=100
STEPS=10000
SAVE_INTERVAL=100
MODEL_NAME='MRI'
NUM_W_LAYERS=14

python -u run.py -sigma $SIGMA -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps 1 \
-learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -mask_type $MASK_TYPE -num_w_layers $NUM_W_LAYERS -mask_type $MASK_TYPE -save_interval $SAVE_INTERVAL #-save_intermediate 

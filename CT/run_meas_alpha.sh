#!/bin/bash
#export CUDA_VISIBLE_DEVICES=5

# I/O arguments
I=1e3
N_ANGLES=25
EPS=25.5 # I_0 = 1000
LEARNING_RATE=0.4
P=0.9
LAMBDA=1e0 # Euclidean loss hyperparameter
LOSS_STR=1*L2+${LAMBDA}*EUCLIDEAN+1*NOISE_L2
echo $LOSS_STR
INPUT_DIR=input_g_0_I_${I}/${N_ANGLES}_views
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_ct/runs/noisy_I_${I}_alpha/${N_ANGLES}_views/stop_morozov_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_ct/runs/noisy_I_${I}_alpha/${N_ANGLES}_views/test_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
OUTPUT_DIR=/scratch/pulse_ct_kl_test_2/p_${P}_lr_${LEARNING_RATE}_reg_${LAMBDA}
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/test
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/test_lr_$lr
#OUTPUT_DIR=/shared/aristotle/MRI/sbhadra/pulse_null_functions/runs/noisy_snr_${SNR}_meas_AR/${MASK_TYPE}/test_bf_alpha
DUPLICATES=10

# PULSE arguments
#EPS=2e-3
#STEPS=10000
STEPS=5000
SAVE_INTERVAL=100

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
#MODEL_NAME='NIH_CT_5000'
MODEL_NAME='NIH_CT_60000'
NUM_W_LAYERS=16

python -u run_alpha.py -n_angles $N_ANGLES -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
-learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_intermediate -save_interval $SAVE_INTERVAL

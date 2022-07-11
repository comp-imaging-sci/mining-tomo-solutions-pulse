#!/bin/bash
#export CUDA_VISIBLE_DEVICES=3

# I/O arguments
IDX=1
I=1e3
LEARNING_RATE=0.4
P=0.001
LAMBDA=0.05 # Euclidean loss hyperparameter
LIMITED_ANGLE=1
RANDOM_FRACTION=0.5
RANDOM_EPOCHS=2000
if [[ $LIMITED_ANGLE -eq 1 ]]
then
    N_PROJ=120
    #N_PROJ=1
    INPUT_DIR=input_airt_y_${IDX}_I_${I}/la_${N_PROJ}
    #OUTPUT_DIR=/shared/radon/MRI/sbhadra/pulse_ct/runs/random_alpha_I_${I}/la_${N_PROJ}/rf_${RANDOM_FRACTION}_re_${RANDOM_EPOCHS}_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
    OUTPUT_DIR=/scratch/test_random_df
else
    N_PROJ=24
    INPUT_DIR=input_airt_y_${IDX}_I_${I}/${N_PROJ}_views
    OUTPUT_DIR=/shared/curie/MRI/sbhadra/pulse_ct/runs/random_alpha_I_${I}/${N_PROJ}_views/real_${IDX}_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
fi
EPS=46432.46 # 120 degrees (I=1000)
MU_MAX=0.063
#MU_MAX=0.0122 # debug
LR_SCHEDULE=linear1cycledrop
LOSS_STR=1*KL+${LAMBDA}*EUCLIDEAN+1*NOISE_L2
echo $LOSS_STR
DUPLICATES=100

# PULSE arguments
STEPS=100
SAVE_INTERVAL=1

# Object and forward operator arguments
MODEL_NAME='NIH_CT_60000'
NUM_W_LAYERS=16

# Don't save intermediate
if [[ $LIMITED_ANGLE -eq 1 ]]
then
    python -u random_run_alpha.py -limited_angle -I $I -n_proj $N_PROJ -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -lr_schedule $LR_SCHEDULE -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
            -mu_max $MU_MAX -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_interval $SAVE_INTERVAL \
            -random_fraction $RANDOM_FRACTION -random_epochs $RANDOM_EPOCHS
else
    python -u run_airt_alpha_kl.py -I $I -n_proj $N_PROJ -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
            -mu_max $MU_MAX -df_constant $DF_CONSTANT -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_interval $SAVE_INTERVAL
fi

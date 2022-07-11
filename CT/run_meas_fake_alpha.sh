#!/bin/bash
#export CUDA_VISIBLE_DEVICES=3

# I/O arguments
IDX=0
I=1e3
LEARNING_RATE=0.4
P=0.001
LAMBDA=0.05 # Euclidean loss hyperparameter
NOISE_LAYERS=9
LIMITED_ANGLE=1
if [[ $LIMITED_ANGLE -eq 1 ]]
then
    N_PROJ=120
    #N_PROJ=1
    INPUT_DIR=input_airt_y_fake_${IDX}_I_${I}/la_${N_PROJ}
    OUTPUT_DIR=/shared/planck/MRI/sbhadra/pulse_ct/runs/fake_alpha_I_${I}/la_${N_PROJ}/noise_layers_${NOISE_LAYERS}_fake_${IDX}_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
    #OUTPUT_DIR=/scratch/pulse_ct_fake/
else
    N_PROJ=24
    INPUT_DIR=input_airt_y_${IDX}_I_${I}/${N_PROJ}_views
    OUTPUT_DIR=/shared/radon/MRI/sbhadra/pulse_ct/runs/noisy_kl_I_${I}_airt_alpha_3/${N_PROJ}_views/real_${IDX}_lr_${LEARNING_RATE}_p_${P}_reg_${LAMBDA}
fi
# DF_CONSTANT=-16305302.15 # 24 views, I_0 = 1e3
# EPS=9308.16 # 24 views, I_0 = 1e3
#DF_CONSTANT=-204351455.70 # 120 degrees, I_0 = 5e3 (obj 2)
#DF_CONSTANT=-208635891.59 # 120 degrees, I_0 = 5e3 (obj 1)
DF_CONSTANT=-42070014.84 # 120 degrees, I_0 = 1e3
#DF_CONSTANT=-417331864.33 # 120 degrees, I_0 = 1e4
#DF_CONSTANT=-36072628802.07 # 1 view, I_0 = 1e8
#DF_CONSTANT=-3606950.21 # 1 view, I_0 = 1e4
#DF_CONSTANT=-82290295.88 # debug
EPS=47640.67 # 120 degrees (I=1000)
#EPS=50688.0 # 120 degrees (tol = 1.1)
#EPS=46540.8 # 120 degrees (tol = 1.01)
#EPS=387.84 # 1 view
MU_MAX=0.063
#MU_MAX=0.0122 # debug
LR_SCHEDULE=linear1cycledrop
#LR_SCHEDULE=rampcosine
LOSS_STR=1*KL+${LAMBDA}*EUCLIDEAN+1*NOISE_L2
echo $LOSS_STR
#OUTPUT_DIR=/scratch/pulse_ct/
DUPLICATES=100

# PULSE arguments
STEPS=10000
SAVE_INTERVAL=50

# Object and forward operator arguments
#MODEL_NAME='MRI_axial_256x256_norm'
#MODEL_NAME='NIH_CT_5000'
MODEL_NAME='NIH_CT_60000'
NUM_W_LAYERS=16

# Save intermediate
# python -u run_alpha_kl.py -I $I -n_angles $N_ANGLES -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
# -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_intermediate -save_interval $SAVE_INTERVAL

# Don't save intermediate
if [[ $LIMITED_ANGLE -eq 1 ]]
then
    python -u run_airt_alpha_kl.py -limited_angle -I $I -n_proj $N_PROJ -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -lr_schedule $LR_SCHEDULE -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
            -mu_max $MU_MAX -df_constant $DF_CONSTANT -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_interval $SAVE_INTERVAL -stop_discrepancy \
            -num_trainable_noise_layers $NOISE_LAYERS
else
    python -u run_airt_alpha_kl.py -I $I -n_proj $N_PROJ -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
            -mu_max $MU_MAX -df_constant $DF_CONSTANT -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_interval $SAVE_INTERVAL
fi


# Limited views
# python -u run_airt_alpha_kl.py -I $I -n_angles $N_ANGLES -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
# -df_constant $DF_CONSTANT -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_interval $SAVE_INTERVAL

# Limited angle
# python -u run_airt_alpha_kl.py -I $I -la $LA -p $P -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -duplicates $DUPLICATES -loss_str $LOSS_STR -loss_domain meas -eps $EPS \
# -df_constant $DF_CONSTANT -learning_rate $LEARNING_RATE -steps $STEPS -model_name $MODEL_NAME -num_w_layers $NUM_W_LAYERS -save_interval $SAVE_INTERVAL -limited_angle $

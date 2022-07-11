#!/bin/bash
OBJ=2
R=6
SIGMA=0.07
#STEP_LR=2e-7
for i in `seq 1`; do
    echo "-----Restart ${i}-----" 
    #python sc_main_no_norm.py +file=nyu_knees
    #python sc_main_no_norm.py +file=step_lr_$STEP_LR
    python sc_main_no_norm.py +file=nyu_knees_${OBJ}_${R}x_sigma_${SIGMA}
done    
echo "Finished all restarts"
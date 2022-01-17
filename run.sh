#!/bin/bash

# training
EPOCHS=50
PATIENCE=10
DEVICE=$1

# optimizer (adam) 
LR=0.0001
BETAS=(0.9, 0.999)
WEIGHT_DECAY=0

# dataset
BATCH_SIZE=64
VAL_SPLIT=0.2

python main.py \
    -val_split $VAL_SPLIT \
    -batch_size $BATCH_SIZE \
    -epochs $EPOCHS \
    -patience $PATIENCE \
    -lr $LR \
    -betas $BETAS \
    -weight_decay $WEIGHT_DECAY \
    -device $DEVICE \
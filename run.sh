#!/usr/bin/env bash

# training
EPOCHS=50
PATIENCE=10
DEVICE=$1

# optimizer  
LR=0.0001

# dataset
BATCH_SIZE=64
VAL_SPLIT=0.2

python main.py \
    -val_split $VAL_SPLIT \
    -batch_size $BATCH_SIZE \
    -epochs $EPOCHS \
    -patience $PATIENCE \
    -lr $LR \
    -device $DEVICE \
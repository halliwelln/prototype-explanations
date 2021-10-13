#!/bin/bash

conda activate kg_env

# ./train.py $1 $2 $3 $4 $5 $6
DATA='mnist'
SHUFFLE_LABELS='yes'
LATENT_DIM=5
NUM_EPOCHS=50
BATCH_SIZE=32
NUM_PROTOTYPES=10
WEIGHT_PATH='/home/nhalliwe/prototype-explanations/weights/'


./train.py $DATA $SHUFFLE_LABELS $LATENT_DIM $NUM_EPOCHS $BATCH_SIZE $NUM_PROTOTYPES $WEIGHT_PATH

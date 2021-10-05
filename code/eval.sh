#!/bin/bash

#conda activate kg_env

DATA='mnist'
LATENT_DIM=5
NUM_PROTOTYPES=10
WEIGHT_PATH='/Users/nhalliwe/Desktop/prototype-explanations/weights/'

./eval.py $DATA $LATENT_DIM $NUM_PROTOTYPES $WEIGHT_PATH

DATA='ca_housing'
LATENT_DIM=2
NUM_PROTOTYPES=4
WEIGHT_PATH='/Users/nhalliwe/Desktop/prototype-explanations/weights/'

./eval.py $DATA $LATENT_DIM $NUM_PROTOTYPES $WEIGHT_PATH

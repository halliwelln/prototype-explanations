#!/usr/bin/env python3

import prototype
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import random as rn
import os
import numpy as np
import utils
from sklearn.metrics import accuracy_score

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('data',type=str)
parser.add_argument('latent_dim',type=int)
parser.add_argument('num_prototypes',type=int)
parser.add_argument('weight_path',type=str)

args = parser.parse_args()

DATA = args.data
LATENT_DIM = args.latent_dim
NUM_PROTOTYPES = args.num_prototypes
WEIGHT_PATH = args.weight_path

if DATA == 'mnist':

    X_train, X_test, y_train, y_test = utils.load_mnist()

    NUM_CLASSES = 10
    IMG_SHAPE = X_train[0].shape
    NUM_FEATURES = X_train[0].shape[0] * X_train[0].shape[1]
    KWARGS = {'img_shape':IMG_SHAPE}

    autoencoder_fun = prototype.image_autoencoder

elif DATA == 'ca_housing':

    X_train, X_test, y_train, y_test, _ = utils.load_ca_housing()

    NUM_CLASSES = 2
    NUM_FEATURES = X_train.shape[-1]
    KWARGS = {'num_features':NUM_FEATURES}

    autoencoder_fun = prototype.tabular_autoencoder

NUM_NEURONS = NUM_FEATURES//2

model = prototype.prototype_model(
    NUM_CLASSES, 
    LATENT_DIM,
    NUM_NEURONS,
    NUM_PROTOTYPES,
    autoencoder_fun,
    **KWARGS
    )

model.load_weights(f'{WEIGHT_PATH}{DATA}.h5')

preds, test_decoded, _ =  model.predict(X_test)
preds = np.argmax(preds,axis=-1)

test_acc = accuracy_score(preds, y_test)

dec_err = np.mean(np.sum(np.square(test_decoded - X_test), axis=1))

print('L2 reconstruction error: {}'.format(round(dec_err, 3)))
print('Accuracy: {}'.format(round(test_acc , 3)))

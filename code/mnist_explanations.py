#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import prototype
import numpy as np
import explain
import pandas as pd
import tensorflow as tf
import utils
import os
import random as rn

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

X_train, X_test, y_train, y_test = utils.load_mnist()

autoencoder_fun = prototype.image_autoencoder

latent_dim = 5
num_neurons = (X_train[0].shape[0] * X_train[0].shape[1]) // 2

mnist_model = prototype.prototype_model(
                    num_classes=10, 
                    latent_dim=latent_dim,
                    num_neurons=num_neurons,
                    num_prototypes=10,
                    autoencoder_fun=autoencoder_fun,
                    **{'img_shape':X_train[0].shape}
                    )

mnist_model.load_weights(os.path.join('..','weights','mnist.h5'))

mnist_encoder = prototype.encoder(mnist_model)

mnist_decoder = prototype.image_decoder(mnist_model, img_shape=(7,7,32))

indices = [0,1,2,3,4,7,8,11,18,61]

arg_sort = np.argsort(y_test[indices])

X_test_subset = X_test[indices][arg_sort]

num_iter = 100
size = (X_test_subset.shape[0],num_iter,7,7,32)
p0 = .1
p1 = .9

diff = explain.explain(tf.convert_to_tensor(X_test_subset),num_iter=num_iter,
                size=size,p0=p0,p1=p1,
                encoder=mnist_encoder,
                decoder=mnist_decoder,seed=SEED)

noisy_model = prototype.prototype_model(
                    num_classes=10, 
                    latent_dim=latent_dim,
                    num_neurons=num_neurons,
                    num_prototypes=10,
                    autoencoder_fun=autoencoder_fun,
                    **{'img_shape':X_train[0].shape}
                    )

noisy_encoder = prototype.encoder(noisy_model)

noisy_decoder = prototype.image_decoder(noisy_model, img_shape=(7,7,32))

num_iter = 100
size = (X_test_subset.shape[0],num_iter,7,7,32)
p0 = .1
p1 = .9

noisy_diff = explain.explain(tf.convert_to_tensor(X_test_subset),num_iter=num_iter,
                size=size,p0=p0,p1=p1,
                encoder=noisy_encoder,
                decoder=noisy_decoder,seed=SEED)

shuffled_model = prototype.prototype_model(
                    num_classes=10, 
                    latent_dim=latent_dim,
                    num_neurons=num_neurons,
                    num_prototypes=10,
                    autoencoder_fun=autoencoder_fun,
                    **{'img_shape':X_train[0].shape}
                    )
shuffled_model.load_weights(os.path.join('..','weights','mnist_shuffled.h5'))

shuffled_encoder = prototype.encoder(shuffled_model)

shuffled_decoder = prototype.image_decoder(shuffled_model, img_shape=(7,7,32))

num_iter = 100
size = (X_test_subset.shape[0],num_iter,7,7,32)
p0 = .1
p1 = .9

shuffled_diff = explain.explain(tf.convert_to_tensor(X_test_subset),num_iter=num_iter,
                size=size,p0=p0,p1=p1,
                encoder=shuffled_encoder,
                decoder=shuffled_decoder,seed=SEED)


for i in arg_sort:
    fig = plt.figure(figsize=(3,3))
    plt.imshow(diff[i,:,:,0],cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join('..','plots',f'rm_heatmap{i}.pdf'),
                bbox_inches='tight', pad_inches=0)   
    plt.close()
    
    fig = plt.figure(figsize=(3,3))
    plt.imshow(X_test_subset[i,:,:,0])
    plt.axis('off')
    plt.savefig(os.path.join('..','plots',f'base_{i}.pdf'),
                bbox_inches='tight', pad_inches=0)  
    plt.close()
    
    fig = plt.figure(figsize=(3,3))
    plt.imshow(noisy_diff[i,:,:,0],cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join('..','plots',f'noisy_heatmap{i}.pdf'),
                bbox_inches='tight', pad_inches=0)   
    plt.close()
    
    fig = plt.figure(figsize=(3,3))
    plt.imshow(shuffled_diff[i,:,:,0],cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join('..','plots',f'shuffled_heatmap{i}.pdf'),
                bbox_inches='tight', pad_inches=0)   
    plt.close()
#!/usr/bin/env python3


if __name__ == "__main__":

    import prototype
    import tensorflow as tf
    import argparse
    import random as rn
    import os
    import utils

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('data',type=str)
    parser.add_argument('latent_dim',type=int)
    parser.add_argument('num_epochs',type=int)
    parser.add_argument('batch_size',type=int)
    parser.add_argument('num_prototypes',type=int)
    parser.add_argument('weight_path',type=str)

    args = parser.parse_args()

    DATA = args.data
    LATENT_DIM = args.latent_dim
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    NUM_PROTOTYPES = args.num_prototypes
    WEIGHT_PATH = args.weight_path

    if DATA == 'mnist':

        X_train, X_test, y_train, y_test = utils.load_mnist()

        y_train_one_hot = tf.keras.utils.to_categorical(y_train)

        NUM_CLASSES = 10
        IMG_SHAPE = X_train[0].shape
        NUM_FEATURES = X_train[0].shape[0] * X_train[0].shape[1]
        KWARGS = {'img_shape':IMG_SHAPE}

    elif DATA == 'ca_housing':

        X_train, X_test, y_train, y_test, _ = utils.load_ca_housing()

        NUM_CLASSES = 2
        NUM_FEATURES = X_train.shape[-1]
        KWARGS = {'num_features':NUM_FEATURES}

        y_train_one_hot = tf.keras.utils.to_categorical(y_train)

    NUM_NEURONS = NUM_FEATURES//2
    LMBDA = 1
    LMBDA_REC = 0.05
    LMBDA_1 = 0.05        
    LMBDA_2 = 0.05
    LEARNING_RATE = 0.0001

    TRAIN_DUMMY = np.zeros((X_train.shape[0], NUM_PROTOTYPES))

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        if DATA == 'mnist':
            autoencoder_fun = prototype.image_autoencoder
        elif DATA == 'ca_housing':
            autoencoder_fun = prototype.tabular_autoencoder

        model = prototype.prototype_model(
            NUM_CLASSES, 
            LATENT_DIM,
            NUM_NEURONS,
            NUM_PROTOTYPES,
            autoencoder_fun,
            **KWARGS
            )
        soft_loss = prototype.softmax_loss(lmbda=LMBDA)

        proto_loss = prototype.prototype_loss(
            lmbda_1=LMBDA_1,
            lmbda_2=LMBDA_2
            )

        dec_loss = prototype.decoder_loss(lmbda_rec=LMBDA_REC)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
            loss={'softmax' : soft_loss, 
                'decoder_output' : dec_loss, 'prototype' : proto_loss},
            metrics=['accuracy']
            )

    history = model.fit(
        X_train, 
        [y_train_one_hot, X_train,TRAIN_DUMMY],
        batch_size=BATCH_SIZE, 
        epochs=NUM_EPOCHS,
        verbose=1,
        shuffle=True
        )

    model.save_weights(f'{WEIGHT_PATH}{DATA}.h5')




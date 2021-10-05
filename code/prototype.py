#!/usr/bin/env python3

from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

class Prototype(tf.keras.layers.Layer):
    
    def __init__(self, n_prototypes, **kwargs):
        super(Prototype, self).__init__(**kwargs)
        self.n_prototypes = n_prototypes

    def build(self, input_shape):

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.n_prototypes, int(input_shape[-1])),
            initializer='glorot_uniform', 
            trainable=True
            )

    def call(self, inputs):

        prototype_distances = distances_l2(inputs, self.kernel)

        return prototype_distances

    def get_config(self):

        config = super(Prototype, self).get_config()
        
        config['n_prototypes'] = self.n_prototypes
        
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def list_of_norms_l2(X):

    square_X = K.square(X)

    return K.sum(square_X, axis=1)

def distances_l2(X, Y):

    XX = K.reshape(list_of_norms_l2(X), shape=(-1,1))
    YY = K.reshape(list_of_norms_l2(Y), shape=(1,-1))

    out = XX + YY - 2 * K.dot(X, K.transpose(Y))

    return out

def softmax_loss(lmbda):    

    def s_loss(y_true, y_pred):

        cross_entropy = lmbda * categorical_crossentropy(y_true, y_pred)
        
        return cross_entropy

    return s_loss

def decoder_loss(lmbda_rec):

    def d_loss(x_true, x_pred):

        rec_loss = lmbda_rec * K.mean(list_of_norms_l2(x_true - x_pred))

        return rec_loss

    return d_loss

def prototype_loss(lmbda_1, lmbda_2):

    def p_loss(proto_true, proto_pred):

        reg_1_loss = lmbda_1 * K.mean(K.min(proto_pred, axis=0))
        #for each prototype vector get min training vector

        reg_2_loss = lmbda_2 * K.mean(K.min(proto_pred, axis=1)) 
        #for teach training example get min prototype vector

        return reg_1_loss + reg_2_loss

    return p_loss 

def tabular_autoencoder(latent_dim,**kwargs):

    num_features = kwargs['num_features']

    encoder_input = tf.keras.layers.Input(shape=(num_features,), name='encoder_input')
    
    encoder_hidden_1 = tf.keras.layers.Dense(
        num_features//2,
        activation='relu', 
        name='encoder_hidden_1')(encoder_input)

    encoded = tf.keras.layers.Dense(
        latent_dim, 
        activation='relu', 
        name='encoder')(encoder_hidden_1)

    decoder_hidden_1 = tf.keras.layers.Dense(
        num_features//2, 
        activation='relu', 
        name='decoder_hidden_1')(encoded)

    decoder_output = tf.keras.layers.Dense(
        num_features,
        activation='sigmoid',
        name='decoder_output')(decoder_hidden_1)

    return encoder_input, encoded, decoder_output

def image_autoencoder(latent_dim, **kwargs):

    img_shape = kwargs['img_shape']

    encoder_input = tf.keras.layers.Input(shape=img_shape,name='encoder_input')

    encoder_hidden_1 = tf.keras.layers.Conv2D(32, 
        kernel_size=(3, 3), 
        activation='relu', 
        padding="same",
        name='encoder_hidden_1')(encoder_input)

    encoder_hidden_pool_1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding='same',
        name='encoder_hidden_pool_1')(encoder_hidden_1)

    encoder_hidden_2 = tf.keras.layers.Conv2D(32, 
        kernel_size=(3, 3), 
        activation='relu', 
        padding='same',
        name='encoder_hidden_2')(encoder_hidden_pool_1)

    encoded = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding='same', 
        name='encoder')(encoder_hidden_2)

    decoder_hidden_1 = tf.keras.layers.Conv2DTranspose(32, 
        kernel_size=(3, 3), 
        strides=2, 
        activation="relu", 
        padding="same",
        name='decoder_hidden_1')(encoded)

    decoder_hidden_2 = tf.keras.layers.Conv2DTranspose(32, 
        kernel_size=(3, 3), 
        strides=2, 
        activation="relu", 
        padding="same",
        name='decoder_hidden_2')(decoder_hidden_1)

    decoder_output = tf.keras.layers.Conv2D(1, 
        kernel_size=(3, 3), 
        activation="sigmoid", 
        padding="same",
        name='decoder_output')(decoder_hidden_2)

    return encoder_input, encoded, decoder_output

def image_decoder(model,img_shape):
    
    # decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    # reshape_layer = tf.keras.layers.Reshape(model.get_layer('encoder').output.shape[1:])
    decoder_input = tf.keras.layers.Input(shape=img_shape, name='decoder_input')

    decoder_layers = [decoder_input]

    for layer in model.layers:
        if 'decoder' in layer.name:
            decoder_layers.append(layer)

    return tf.keras.Sequential(decoder_layers)

def tabular_decoder(model, latent_dim):

    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')

    decoder_layers = [decoder_input]

    for layer in model.layers:
        if 'decoder' in layer.name:
            decoder_layers.append(layer)

    return tf.keras.Sequential(decoder_layers)

def encoder(model):
    return tf.keras.Model(
        inputs=model.input, 
        outputs=model.get_layer('encoder').output
        )

def prototype_model(
    num_classes, 
    latent_dim,
    num_neurons,
    num_prototypes,
    autoencoder_fun,
    **kwargs
    ):

    '''
    Implementation of Prototype Network
    https://arxiv.org/abs/1710.04806
    Parameters
    ----------

    num_classes: int
        number of classes
    num_prototypes: int
        number of prototype vectors
    autoencoder_fun: function
        autoencoder function returning input layer,encoded layer,decoder layer
    Returns
    ----------
    model: keras model
        final prototype model including autoencoder
    '''

    encoder_input, encoded, decoder_output = autoencoder_fun(
                                                    latent_dim,
                                                    **kwargs
                                                    )

    flatten = tf.keras.layers.Flatten()(encoded)

    prototype_distances = Prototype(
        num_prototypes, 
        name='prototype'
        )(flatten)

    fully_connected = tf.keras.layers.Dense(
        num_neurons,
        activation='relu',
        name='fully_connected',
        )(prototype_distances)

    output = tf.keras.layers.Dense(
        num_classes,
        name='softmax',
        activation='softmax',
        )(fully_connected)

    model = tf.keras.models.Model(
        inputs=encoder_input, 
        outputs=[
            output, 
            decoder_output,
            prototype_distances
        ]
    )

    return model
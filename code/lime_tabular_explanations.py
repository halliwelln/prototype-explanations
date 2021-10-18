#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import lime.lime_tabular
import utils
import prototype
import os
import explain
import tensorflow as tf
import random as rn
import numpy as np
import pandas as pd

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

def pred(x):
    return model.predict(x)[0]

X_train, X_test, y_train, y_test, names = utils.load_ca_housing()

autoencoder_fun = prototype.tabular_autoencoder

latent_dim = 2

model = prototype.prototype_model(
                    num_classes=2, 
                    latent_dim=latent_dim,
                    num_neurons=X_train.shape[-1]//2,
                    num_prototypes=4,
                    autoencoder_fun=autoencoder_fun,
                    **{'num_features':X_train.shape[-1]}
                    )

model.load_weights(os.path.join('..','weights','ca_housing.h5'))

housing_encoder = prototype.encoder(model)
housing_decoder = prototype.tabular_decoder(model, latent_dim=latent_dim)

num_iter = 100
size = (X_test.shape[0], num_iter,latent_dim)
p0 = .1
p1 = .9

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=names, class_names=['Below median','Above median'])

diff = explain.explain(tf.convert_to_tensor(X_test),num_iter=num_iter,
                size=size,p0=p0,p1=p1,
                encoder=housing_encoder,
                decoder=housing_decoder,seed=SEED)


for i in [26,57]:

    exp = explainer.explain_instance(X_test[i], pred, num_features=8)
    lime_scores = sorted(list(exp.__dict__['local_exp'].values())[0], key=lambda x: abs(x[1]), reverse=True)
    lime_scores = [(names[tup[0]], abs(tup[1])) for tup in lime_scores]
    
    lime_df = pd.DataFrame(lime_scores, columns=['Feature', 'Score']).sort_values(by='Feature', ascending=False)
    
    y_pos = np.arange(len(lime_df))
    
    fig, ax = plt.subplots()

    ax.barh(lime_df.iloc[:,0], lime_df.iloc[:,1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lime_df.iloc[:,0],fontsize=15)

    ax.set_xticks([])
    plt.savefig(os.path.join('..','plots',f'lime_features_{i}.pdf'),
                     bbox_inches='tight', pad_inches=0)  
    plt.close()
    
    b = pd.DataFrame(zip(names, diff[i]),columns=['Feature', 'Score']).sort_values(by='Feature', ascending=False)
    
    fig, ax = plt.subplots()

    ax.barh(b.iloc[:,0], b.iloc[:,1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(b.iloc[:,0],fontsize=15)
    ax.set_xticks([])
    plt.savefig(os.path.join('..','plots',f'proposed_features_{i}.pdf'),
                    bbox_inches='tight', pad_inches=0)   
    plt.close()

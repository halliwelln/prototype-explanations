#!/usr/bin/env python3

import lime
import lime.lime_tabular
import utils
import prototype
import os
import json
import pandas as pd

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

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=names, class_names=[0,1])

counts = {}

for i in range(len(X_test)):
    
    exp = explainer.explain_instance(X_test[i], pred, num_features=8,top_labels=1)
    
    dim = str(sorted(list(exp.__dict__['local_exp'].values())[0],
        key=lambda x: abs(x[1]), reverse=True)[0][0])
    
    if dim in counts:
        counts[dim] += 1
    else:
        counts[dim] = 1

with open(os.path.join('..','data','lime_dim_counts.json'),"w") as f:

    json.dump(counts, f)

print('Done.')
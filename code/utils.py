#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(array):
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

def load_mnist():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    return X_train, X_test, y_train, y_test

def load_ca_housing():

    cal_housing = fetch_california_housing()

    X, y = cal_housing.data, cal_housing.target
    names = cal_housing.feature_names
    X = pd.DataFrame(X, columns=names)
    y = (y>2.5).astype(int)

    mm_scaler = MinMaxScaler()
    X = mm_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=.33, 
            random_state=42
            )

    return X_train, X_test, y_train, y_test, names

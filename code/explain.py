#!/usr/bin/env python3

import numpy as np

def explain(X_test,num_iter,size,p0,p1,encoder,decoder,seed):
    
    rng = np.random.RandomState(seed)
    rand = rng.choice([0, 1], size=size, p=[p0,p1])
    
    x_test_encoded = encoder.predict(X_test)

    encoded_repeat = np.repeat(np.expand_dims(x_test_encoded,axis=1),num_iter,axis=1)

    mask = (encoded_repeat * rand).mean(axis=1)
    
    rand_decoded = decoder.predict(mask)

    x_test_decoded = decoder.predict(x_test_encoded)
    
    diff = np.abs(x_test_decoded - rand_decoded)
    
    return diff

if __name__ == "__main__":

    explain(X_test,size,p,encoder,decoder,seed)
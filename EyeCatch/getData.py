import numpy as np
import os

def get_data():
    """only get the first dir"""
    root_dirs = os.listdir('../data')

    # x
    for root in root_dirs:
        X = np.load('../data/' + root + '/' + root + '-x.npy')
        Y = np.load('../data/' + root + '/' + root + '-y.npy')
        break

    return X, Y

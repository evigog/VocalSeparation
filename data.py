import numpy as np

def load_batch():
    return np.load('dataset/batch_mixed4.npy'), np.load('dataset/batch_vc4.npy')

import numpy as np

def load_batch():
    temp_train = np.load('coefficients/batch_stft_train.npz')
    temp_val = np.load('coefficients/batch_stft_dev.npz')
    return temp_train["mixed"].tolist(), temp_train["vc"].tolist(), temp_val["mixed"].tolist(), temp_val["vc"].tolist()

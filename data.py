import numpy as np

def load_batch():
    temp = np.load('coefficients/batch_stft_train.npz')
    return temp["mixed"].tolist(), temp["vc"].tolist()

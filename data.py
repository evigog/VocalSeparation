import numpy as np

def load_batch(feature): #mfcc or stft

    if feature == 'stft':
        temp_train = np.load('coefficients/batch_stft_train.npz')
        temp_val = np.load('coefficients/batch_stft_dev.npz')
    elif feature =='mfcc':
        temp_train = np.load('coefficients/batch_mfcc_train.npz') #mfccs are used!
        temp_val = np.load('coefficients/batch_mfcc_dev.npz')
    else:
        print("mfcc or stft - try again!")
        return -1

    return temp_train["mixed"].tolist(), temp_train["vc"].tolist(), temp_val["mixed"].tolist(), temp_val["vc"].tolist()

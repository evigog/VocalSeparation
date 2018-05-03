from prep_tools import *
from scipy.io import wavfile
import os


def load_dataset(n_samples):
    cache_path = 'E:/data/karaoke/'
    cache_filename = 'data_train.npz'
    path_dataset = 'dataset/Wavfile'  # all songs are sampled at 16kHz
    if not os.path.isfile(cache_path + str(n_samples) + cache_filename):
        samples = [wavfile.read(os.path.join(path_dataset, f))[1] for f in os.listdir(path_dataset)
                    if os.path.isfile(os.path.join(path_dataset, f))] [:n_samples]
        np.savez_compressed(cache_path + str(n_samples) + cache_filename, samples=samples)
        print('Saved compressed dataset to cache.')
    else:
        data_train = np.load(cache_path + str(n_samples) + cache_filename)
        samples = data_train['samples'].tolist()
        print('Loaded compressed dataset from cache.')
    mfcc_mixed = [mfcc(np.sum(smp, axis=-1)) for smp in samples]
    mfcc_bg = [mfcc(smp[:,0]) for smp in samples]
    mfcc_vc = [mfcc(smp[:,1]) for smp in samples]
    return samples, mfcc_mixed, mfcc_vc, mfcc_bg


## TODO: data_augmentation()


# >>>>>>>START

load_dataset(172)

from .prep_tools import *
from scipy.io import wavfile
from .Parameters import *
import os

def remove_dirty(data):
    #compute avg after removing nan
    not_nan = data[np.isfinite(data)]
    avg_value = np.mean(not_nan)

    #replace nan with the mean value of current song
    remove_nan = data
    remove_nan[np.isnan(data)] = avg_value

    return remove_nan

def create_dataset(n_samples):
    cache_path = 'data/temp/'
    cache_filename = 'data_train.npz'
    path_dataset = 'data/Wavfile'  # all songs are sampled at 16kHz

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


    clear_mfcc_mixed = [remove_dirty(song) for song in mfcc_mixed]
    clear_mfcc_bg = [remove_dirty(song) for song in mfcc_bg]
    clear_mfcc_vc = [remove_dirty(song) for song in mfcc_vc]

    print(np.isnan(clear_mfcc_vc[35]).any())


    np.save('dataset/mfcc_mixed', clear_mfcc_mixed)
    np.save('dataset/mfcc_bg', clear_mfcc_bg)
    np.save('dataset/mfcc_vc', clear_mfcc_vc)

    #split coef matrices into batches
    batch_mixed = [coef_to_batch(src_mixed) for src_mixed in mfcc_mixed]
    batch_bg = [coef_to_batch(src_bg) for src_bg in mfcc_bg]
    batch_vc = [coef_to_batch(src_vc) for src_vc in mfcc_vc]

    print(np.isnan(batch_vc[35]).any())

    np.save('dataset/batch_mixed4', batch_mixed)
    np.save('dataset/batch_bg4', batch_bg)
    np.save('dataset/batch_vc4', batch_vc)


def coef_to_batch(src): #output shape: (num_batches, num_frames, num_coef)

    num_frames = src.shape[0]
    num_coef = src.shape[1]

    #fill last batch with zeros if necessary
    if (num_frames % Preprocessing.MAX_TIME != 0 ):
        mod = Preprocessing.MAX_TIME - (num_frames % Preprocessing.MAX_TIME)
        z = np.zeros((1, int(mod) * num_coef)).reshape((-1, num_coef))
        src = np.vstack((src, z))

    batches = np.reshape(src, (-1, Preprocessing.MAX_TIME, Preprocessing.NUM_COEF))

    return batches




## TODO: data_augmentation()

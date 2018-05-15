from preprocessing.prep_tools import *
from scipy.io import wavfile
from preprocessing.Parameters import *
import os


def load_dataset(n_samples):
    cache_path = '/Users/evi/Documents/KTH_ML/Period_4/Speech_Recognition/Project/temp'
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

    clear_mfcc_mixed = [remove_dirty(song) for song in mfcc_mixed]
    clear_mfcc_bg = [remove_dirty(song) for song in mfcc_bg]
    clear_mfcc_vc = [remove_dirty(song) for song in mfcc_vc]

    np.save('coefficients/mfcc_mixed', clear_mfcc_mixed)
    np.save('coefficients/mfcc_bg', clear_mfcc_bg)
    np.save('coefficients/mfcc_vc', clear_mfcc_vc)

    #split coef matrices into batches
    batch_mixed = [coef_to_batch(src_mixed) for src_mixed in clear_mfcc_mixed]
    batch_bg = [coef_to_batch(src_bg) for src_bg in clear_mfcc_bg]
    batch_vc = [coef_to_batch(src_vc) for src_vc in clear_mfcc_vc]

    np.save('coefficients/batch_mixed4', batch_mixed)
    np.save('coefficients/batch_bg4', batch_bg)
    np.save('coefficients/batch_vc4', batch_vc)

    return samples, batch_mixed, batch_bg, batch_vc

#take care of nan or inf values
def remove_dirty(data):

    #compute avg after removing nan
    not_nan = data[np.isfinite(data)]
    avg_value = np.mean(not_nan)

    #replace nan with the mean value of current song
    remove_nan = data
    remove_nan[np.isnan(data)] = avg_value

    return remove_nan


#transform matrix coefficient form to batches
#computed for one song
def coef_to_batch(src): #output shape: (num_batches, num_frames, num_coef)

    num_frames = src.shape[0]
    num_coef = src.shape[1]

    #fill last batch with zeros if necessary
    if (num_frames % Preprocessing.BATCH_SIZE != 0 ):
        mod = Preprocessing.BATCH_SIZE - (num_frames % Preprocessing.BATCH_SIZE)
        z = np.zeros((1, int(mod) * num_coef)).reshape((-1, num_coef))
        src = np.vstack((src, z))  #apppend zeros in the end

    batches = np.reshape(src, (-1, Preprocessing.BATCH_SIZE, Preprocessing.NUM_COEF))

    return batches

#transform batches to matrix coefficient form
#computed for one song
def batch_to_coef(batches, original_frames):  #output shape: (num_frames, num_coef)

   num_batches, num_frames, num_coefs = batches.shape

   coefs = np.reshape(batches, (-1, num_coefs))
   #remove padding lines
   without_padding = coefs[0:original_frames]

   return without_padding






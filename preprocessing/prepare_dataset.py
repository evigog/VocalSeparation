from .prep_tools import *
from .Parameters import *
from constants import *

import os
import random as rand

from pathlib import Path

def create_all_dataset(n_files):
    path_trainset = Path('data/Wavfile/train')

    if not(path_trainset.is_dir()):
        print("Spliting the dataset into training, validation and testing set...")
        split_dataset()


    n_files_train = int(n_files * TRAINING_SPLIT)
    n_files_test = int(n_files * (TEST_SPLIT))
    n_files_dev = n_files - (n_files_train + n_files_test)
    print("Creating trainset...", n_files_train)
    create_dataset("train", n_files_train)
    print("Creating devset...", n_files_dev)
    create_dataset("dev", n_files_dev)
    print("Creating testset...", n_files_test)
    create_dataset("test", n_files_test)



#load_path: train or test  (folder of .wav files to process)
#n_samples: number of .wav files to process
def create_dataset(load_path, n_files):

    path_dataset = 'data/Wavfile' + '/' + load_path # all songs are sampled at 16kHz
    if not os.path.isfile('coefficients/batch_stft_%s.npz' %load_path):
        filenames = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset)
                    if os.path.isfile(os.path.join(path_dataset, f))][:n_files]

        # Not efficient, it's better to load all the files at once

        #produce Short Time Fourier Transform - returns a list of lists (magnitude, phase)
        stft_mixed = [wav_to_stft(f, channel='mixed') for f in filenames]
        stft_bg = [wav_to_stft(f, channel='instrumental') for f in filenames]
        stft_vc = [wav_to_stft(f, channel='vocals') for f in filenames]

        ###normalization step
        stft_mixed_norm = [np.divide(src_mixed[0],100) for src_mixed in stft_mixed]
        stft_bg_norm = [np.divide(src_bg[0], 100) for src_bg in stft_bg]
        stft_vc_norm = [np.divide(src_vc[0], 100) for src_vc in stft_vc]

	##taking logarithms of STFTs
        log_stft_mixed_norm = [np.log(src_mx+Preprocessing.eps) for src_mx in stft_mixed_norm]
        log_stft_bg_norm = [np.log(src_bg+Preprocessing.eps) for src_bg in stft_bg_norm]
        log_stft_vc_norm = [np.log(src_vc+Preprocessing.eps) for src_vc in stft_vc_norm]

        #produce MFCC coefficients
        mfcc_mixed = [wav_to_mfcc(f, channel='mixed') for f in filenames]
        mfcc_bg = [wav_to_mfcc(f, channel='instrumental') for f in filenames]
        mfcc_vc = [wav_to_mfcc(f, channel='vocals') for f in filenames]
	
        mins_mx, mins_bg, mins_vc = [], [], []
        for mm, mb, mv in zip(mfcc_mixed, mfcc_bg, mfcc_vc):
            mins_mx.append(np.min(mm))
            mins_bg.append(np.min(mb))
            mins_vc.append(np.min(mv))
        absmin_mx, absmin_bg, absmin_vc = abs(np.min(mins_mx)), abs(np.min(mins_bg)), abs(np.min(mins_vc))
        ##taking logarithms of MFCCs
        log_mfcc_mixed = [np.log(src_mx+absmin_mx+Preprocessing.eps) for src_mx in mfcc_mixed]
        log_mfcc_bg = [np.log(src_bg+absmin_bg+Preprocessing.eps) for src_bg in mfcc_bg]
        log_mfcc_vc = [np.log(src_vc+absmin_vc+Preprocessing.eps) for src_vc in mfcc_vc]	

        #save only the magnitude (stft)
        #np.savez_compressed('coefficients/stft_%s.npz' %load_path, mixed=stft_mixed ,bg=stft_bg, vc=stft_vc)
        #np.savez_compressed('coefficients/mfcc_%s.npz' %load_path, mixed=mfcc_mixed ,bg=mfcc_bg, vc=mfcc_vc)

        #split coef matrices into batches

        #STFT - no normalization
        #batch_stft_mixed = [coef_to_batch(src_mixed[0]) for src_mixed in stft_mixed]
        #batch_stft_bg = [coef_to_batch(src_bg[0]) for src_bg in stft_bg]
        #batch_stft_vc = [coef_to_batch(src_vc[0]) for src_vc in stft_vc]

        #STFT - with normalization
        batch_stft_mixed = [coef_to_batch(src_mixed) for src_mixed in log_stft_mixed_norm]
        batch_stft_bg = [coef_to_batch(src_bg) for src_bg in  log_stft_bg_norm]
        batch_stft_vc = [coef_to_batch(src_vc) for src_vc in log_stft_vc_norm]


        #MFCC
        batch_mfcc_mixed = [coef_to_batch(src_mixed) for src_mixed in log_mfcc_mixed]
        batch_mfcc_bg = [coef_to_batch(src_bg) for src_bg in log_mfcc_bg]
        batch_mfcc_vc = [coef_to_batch(src_vc) for src_vc in log_mfcc_vc]

        np.savez_compressed('coefficients/batch_stft_%s.npz' %load_path, mixed=batch_stft_mixed ,bg=batch_stft_bg, vc=batch_stft_vc)
        np.savez_compressed('coefficients/batch_mfcc_%s.npz' %load_path, mixed=batch_mfcc_mixed ,bg=batch_mfcc_bg, vc=batch_mfcc_vc)
        print('Saved coefficients.')


#split dataset into training. validation and testing
#create /train, /dev and /test subfolders inside folder data/Wavfile
def split_dataset():
    path_dataset = 'data/Wavfile'
    filenames = os.listdir(path_dataset)  #names of all .wav files
    rand.seed(230)
    rand.shuffle(filenames)  # shuffles the ordering of filenames (deterministic given the chosen seed)

    train_lim = int(TRAINING_SPLIT * len(filenames))
    test_lim = int((1 - TEST_SPLIT) * len(filenames))

    train_filenames = filenames[: train_lim]
    test_filenames = filenames[test_lim:]
    dev_filenames = filenames[train_lim:test_lim]


    train_path = os.path.join(path_dataset, 'train')
    test_path = os.path.join(path_dataset, 'test')
    dev_path = os.path.join(path_dataset, 'dev')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(dev_path):
        os.makedirs(dev_path)

    #move training files to data/Wavefile/train folder
    for f in train_filenames:
      os.rename(os.path.join(path_dataset, f), os.path.join(train_path, f))

    # move files to data/Wavefile/test folder
    for f in test_filenames:
        os.rename(os.path.join(path_dataset, f), os.path.join(test_path, f))

    # move files to data/Wavefile/dev folder
    for f in dev_filenames:
        os.rename(os.path.join(path_dataset, f), os.path.join(dev_path, f))

#take care of nan or inf values
def remove_dirty(data):  # SHOULD NOT BE NECESSARY AFTER MFCC FIX

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

    batches = np.reshape(src, (-1, Preprocessing.BATCH_SIZE, num_coef))

    return batches

#transform batches to matrix coefficient form
#computed for one song
def batch_to_coef(batches, original_frames=409):  #output shape: (num_frames, num_coef)

   num_batches, num_frames, num_coefs = batches.shape

   coefs = np.reshape(batches, (-1, num_coefs))
   #remove padding lines
   # without_padding = coefs[0:original_frames]  DISCUSS 16/05/2018

   # return without_padding   DISCUSS 16/05/2018
   return coefs

from .prep_tools import *
from .Parameters import *
from constants import *

import os
import random as rand

from pathlib import Path

def create_all_dataset(n_files):
    path_trainset = Path('data/Wavfile/train')
    path_testset = Path('data/Wavfile/test')

    if not(path_trainset.is_dir()):
        print("Spliting the dataset into training and validation set...")
        split_dataset()


    n_files_train = int(n_files * TRAINING_SPLIT)
    n_files_test = int(n_files * (1 - TRAINING_SPLIT))
    print("Creating trainset...", n_files_train)
    create_dataset("train", n_files_train)
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

        #produce Short Time Fourier Transform
        stft_mixed = [wav_to_stft(f, channel='mixed') for f in filenames]
        stft_bg = [wav_to_stft(f, channel='instrumental') for f in filenames]
        stft_vc = [wav_to_stft(f, channel='vocals') for f in filenames]

        #produce MFCC coefficients
        mfcc_mixed = [wav_to_mfcc(f, channel='mixed')[0] for f in filenames]
        mfcc_bg = [wav_to_mfcc(f, channel='instrumental')[0] for f in filenames]
        mfcc_vc = [wav_to_mfcc(f, channel='vocals')[0] for f in filenames]

        np.savez_compressed('coefficients/stft_%s.npz' %load_path, mixed=stft_mixed ,bg=stft_bg, vc=stft_vc)
        np.savez_compressed('coefficients/mfcc_%s.npz' %load_path, mixed=mfcc_mixed ,bg=mfcc_bg, vc=mfcc_vc)

        #split coef matrices into batches

        #STFT
        batch_stft_mixed = [coef_to_batch(src_mixed) for src_mixed in stft_mixed]
        batch_stft_bg = [coef_to_batch(src_bg) for src_bg in stft_bg]
        batch_stft_vc = [coef_to_batch(src_vc) for src_vc in stft_vc]
        #MFCC
        batch_mfcc_mixed = [coef_to_batch(src_mixed) for src_mixed in mfcc_mixed]
        batch_mfcc_bg = [coef_to_batch(src_bg) for src_bg in mfcc_bg]
        batch_mfcc_vc = [coef_to_batch(src_vc) for src_vc in mfcc_vc]

        np.savez_compressed('coefficients/batch_stft_%s.npz' %load_path, mixed=batch_stft_mixed ,bg=batch_stft_bg, vc=batch_stft_vc)
        np.savez_compressed('coefficients/batch_mfcc_%s.npz' %load_path, mixed=batch_mfcc_mixed ,bg=batch_mfcc_bg, vc=batch_mfcc_vc)
        print('Saved coefficients.')


#split dataset into training and testing
#create /train and /test subfolders inside folder data/Wavfile
def split_dataset():
    path_dataset = 'data/Wavfile'
    filenames = os.listdir(path_dataset) #names of all .wav files
    rand.seed(230)
    rand.shuffle(filenames)  # shuffles the ordering of filenames (deterministic given the chosen seed)

    split_1 = int(TRAINING_SPLIT * len(filenames))  #training set
    train_filenames = filenames[:split_1]
    test_filenames = filenames[split_1:]

    # no validation set used!
    # split_2 = int((1 - TRAINING_SPLIT) * len(filenames))
    # dev_filenames = filenames[split_1:split_2]

    train_path = os.path.join(path_dataset, 'train')
    test_path = os.path.join(path_dataset, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    #move training files to data/Wavefile/train folder
    for f in train_filenames:
      os.rename(os.path.join(path_dataset, f), os.path.join(train_path, f))

    # move training files to data/Wavefile/test folder
    for f in test_filenames:
        os.rename(os.path.join(path_dataset, f), os.path.join(test_path, f))

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

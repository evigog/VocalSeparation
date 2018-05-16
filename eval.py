import network
from constants import *
from preprocessing.prep_tools import  *
from preprocessing.prepare_dataset import coef_to_batch, batch_to_coef

import tensorflow as tf
import argparse
import mir_eval  #for evaluating the constracted voice
import os
import numpy as np


def loadSong():
    #load song from data/wavefile/test
    test_path = 'data/Wavfile/test'
    test_filenames = os.listdir(test_path)

    print('Choose a random song for testing')
    indx = np.random.randint(0, len(test_filenames)-1)  #take a random test song

    song_filename = os.path.join(test_path, test_filenames[indx])

    song_stft_mixed = wav_to_stft(song_filename, channel='mixed')
    batches_stft_mixed = coef_to_batch(song_stft_mixed)

    X = batches_stft_mixed #input features to network
    Y = read_wavfile(song_filename, channel='vocals')

    print('input to network is ready!')

    return X, Y


def predict():
    net = network.RNN_network()

    print("Start eval")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        net.load_state(sess, CKPT_PATH) #load trained model

        X, Y = loadSong()

        predictions = sess.run(  #coefficients that correspond to voice
            [net()],
            feed_dict={
                net.batchX_placeholder:X,
                net.batchY_placeholder:Y
            })

    #restore the coefficients from network output(batch_to_coef function)
    #restore song from coefficients
    #call evaluate_voice: returns a dict of evaluation metrics




#reference_voice, predicted_voice are the source signals (ndarrays of samples)
def evaluate_voice(reference_voice, predicted_voice):

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(reference_voice, predicted_voice)

    eval = {'sdr':sdr, 'sir':sir, 'sar':sar, 'perm':perm}

    return eval




if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    args = parser.parse_args()

    loadSong()

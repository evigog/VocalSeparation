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

    #mixed signal
    (song_stft_mixed_magn, song_stft_mixed_phase) = wav_to_stft(song_filename, channel='mixed')
    batches_stft_mixed_magn = coef_to_batch(song_stft_mixed_magn)

    #vocal signal
    (song_stft_vocal_magn, song_stft_vocal_phase) = wav_to_stft(song_filename, channel='vocals')
    batches_stft_vocal_magn = coef_to_batch(song_stft_vocal_magn)

    X = batches_stft_mixed_magn #magnitude mixed coefficients in batches (network input)
    Y = batches_stft_vocal_magn    #magnitude vocal coefficients in batches (network output)
    original_vocal_wav = read_wavfile(song_filename, channel='vocals') #vocals wav

    #save_audio_to_file(original_vocal_wav, "original.wav")

    print('input to network is ready!')

    return X, Y, original_vocal_wav, song_stft_mixed_phase      #song_stft_mixed_phase


def predict():
    net = network.RNN_network()

    print("Start evaluation")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        net.load_state(sess, CKPT_PATH) #load trained model

        X, Y, original_vocals_wav, mixed_phase = loadSong()

        predict_vocal_magnitude_batch = sess.run(  #coefficients that correspond to voice (in a list)
            [net()],
            feed_dict={
                net.batchX_placeholder:X,
                net.batchY_placeholder:Y
            })

    #restore the magnitude coefficients from network output(batch_to_coef function)
    predict_coef_magn = batch_to_coef(predict_vocal_magnitude_batch[0])
    #reconstruct wav signal
    original_num_frames = mixed_phase.shape[0]  #remove padding
    reconstructed_vocal = stft_to_wav(predict_coef_magn[0:original_num_frames], mixed_phase)
    save_audio_to_file(reconstructed_vocal, "reconstruct.wav")
    eval = evaluate_voice(original_vocals_wav, reconstructed_vocal)

    print("sdr: ", eval["sdr"], " sir: ", eval["sir"], " sar: ", eval["sar"], " perm: ", eval["perm"])




#reference_voice, predicted_voice are the source signals (ndarrays of samples)
def evaluate_voice(reference_voice, predicted_voice):

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(reference_voice, predicted_voice)

    eval = {'sdr':sdr, 'sir':sir, 'sar':sar, 'perm':perm}

    return eval




if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    args = parser.parse_args()

    predict()

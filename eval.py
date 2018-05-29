import network
from constants import *
from preprocessing.prep_tools import  *
from preprocessing.prepare_dataset import coef_to_batch, batch_to_coef

import tensorflow as tf
import argparse
import mir_eval  #for evaluating the constracted voice
import os
import numpy as np
from pathlib import Path


def loadSong(num_files, feature):
    #load song from data/wavefile/test
    test_path = 'data/Wavfile/test'
    test_filenames = os.listdir(test_path)

    print('Choose random songs for testing')
    indx_list = np.random.randint(0, len(test_filenames)-1, num_files).tolist()  #take a random test song

    song_filename_list = [os.path.join(test_path, test_filenames[indx]) for indx in indx_list]

    original_vocal_wavs = [read_wavfile(song_filename, channel='vocals') for song_filename in song_filename_list]  # vocals wav


    if (feature == 'stft'):

         #mixed signal #list (magnitude, phase)
         song_stft_mixed = [wav_to_stft(song_filename, channel='mixed') for song_filename in song_filename_list]

         #vocal signal
         song_stft_vocal = [wav_to_stft(song_filename, channel='vocals') for song_filename in song_filename_list]

         song_features_mixed = [song[0] for song in song_stft_mixed]
         song_features_vocals = [song[0] for song in song_stft_vocal]
         song_vocals_phase = [song[1] for song in song_stft_vocal] #used to reconstruct the vocal signal

    else: #mfcc used

        # mixed signal
        song_mfcc_mixed = [wav_to_mfcc(song_filename, channel='mixed') for song_filename in song_filename_list]

        # vocal signal
        song_mfcc_vocal = [wav_to_mfcc(song_filename, channel='vocals') for song_filename in song_filename_list]

        song_features_mixed = [song for song in song_mfcc_mixed]
        song_features_vocals = [song for song in song_mfcc_vocal]
        song_vocals_phase = -1


    num_original_frames = [song_vocal.shape[0] for song_vocal in song_features_vocals]
    batches_mixed = [coef_to_batch(song_mixed) for song_mixed in song_features_mixed]
    batches_vocals = [coef_to_batch(song_vocal) for song_vocal in song_features_vocals]

    X = batches_mixed     #magnitude mixed coefficients in batches (network input)
    Y = batches_vocals    #magnitude vocal coefficients in batches (network output)

    original_instrumental_wavs = [read_wavfile(song_filename, channel='instrumental') for song_filename in song_filename_list] #instrumental wav used for evaluation

    print('input to network is ready!')

    return X, Y, original_vocal_wavs, song_vocals_phase, original_instrumental_wavs, num_original_frames


def predict(num_files, feature):
    net = network.RNN_network()

    print("Start evaluation")

    sdr_list = []
    sir_list = []
    sar_list = []
    weights_list = []

    wav_path = Path(os.path.join(SAVE_PATH, 'no_dropout/vocal_wavs'))
    if not (wav_path.is_dir()):
        os.mkdir(wav_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        net.load_state(sess, CKPT_PATH) #load trained model


        X, Y, original_vocals_wavs, vocal_phases, original_instrumental_wavs, original_num_frames_list = loadSong(num_files, feature)  #lists
        #original_num_frames = list with the original num of frames for each song

        num_tests = len(X)

        for i in range(num_tests):

            predict_vocal_magnitude_batch = sess.run(  #coefficients that correspond to voice (in a list)
                [net()],
                feed_dict={
                    net.batchX_placeholder:X[i],
                    net.batchY_placeholder:Y[i]
                })

            #restore the magnitude coefficients from network output(batch_to_coef function)
            predict_coef_magn = batch_to_coef(predict_vocal_magnitude_batch[0])

            #*****Griffinlim reconstruction*****
            original_num_frames = original_num_frames_list[i] #remove padding frames
            if feature == 'mfcc':
                reconstructed_vocal = reconstruct_signal_griffin_lim(predict_coef_magn[:,:31], 60, 30, 201)[0:original_num_frames]
            else: #stft
                reconstructed_vocal = reconstruct_signal_griffin_lim(predict_coef_magn * 100, 512, 256, 201)[0:original_num_frames]


            # save predicted vocal wav
            #save_audio_to_file(reconstructed_vocal,  os.path.join(wav_path, 'out_%i.wav' %i))

            evaluate_len = reconstructed_vocal.shape[0]
            #if evaluated vocal component is all zeros, then ignore song
            if (np.count_nonzero(original_vocals_wavs[i][0:evaluate_len]) != 0):
                original_components = np.vstack((original_vocals_wavs[i][0:evaluate_len], original_instrumental_wavs[i][0:evaluate_len]))
                reconstructed_vocal_stacked = np.vstack((reconstructed_vocal, reconstructed_vocal))

                eval = evaluate_voice(original_components, reconstructed_vocal_stacked)

                print("sdr: ", eval["sdr"][0], " sir: ", eval["sir"][0], " sar: ", eval["sar"][0])

                sdr_list.append(eval["sdr"][0])
                sir_list.append(eval["sir"][0])
                sar_list.append(eval["sar"][0])
                weights_list.append(original_vocals_wavs[i].size)


    #compute global evaluation metric, weighted average using song length
    global_sdr = np.average(np.asarray(sdr_list), axis=0, weights=np.asarray(weights_list))
    global_sar = np.average(np.asarray(sar_list), axis=0, weights=np.asarray(weights_list))
    global_sir = np.average(np.asarray(sir_list), axis=0, weights=np.asarray(weights_list))
    print("Global sdr: ",  global_sdr, " Global sar: ",  global_sar, "Global sir: ",  global_sir)


#reference_voice, predicted_voice are the source signals (ndarrays of samples)
def evaluate_voice(reference_voice, predicted_voice):

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(reference_voice, predicted_voice)

    eval = {'sdr':sdr, 'sir':sir, 'sar':sar, 'perm':perm}

    return eval




if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    args = parser.parse_args()

    predict(20, 'mfcc')

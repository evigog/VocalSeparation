import network
from constants import *
from preprocessing.prep_tools import  read_wavfile

import tensorflow as tf
import argparse
import mir_eval  #for evaluating the constracted voice


def loadSong():
    #load song from data/wavefile/test
    # (X, Y) <- convert song to batch batcoefficients


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

    predict()

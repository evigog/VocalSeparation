import network
from constants import *
import numpy as np
import data
import tensorflow as tf
import argparse
import os
import shutil

def loadSong():
    X, Y = 0, 0
    return X, Y


def train():
    net = network.RNN_network()

    print("Start eval")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        net.load_state(sess, CKPT_PATH)

        X, Y = loadSong()

        _output = sess.run(
            [net()],
            feed_dict={
                net.batchX_placeholder:X,
                net.batchY_placeholder:Y
            })


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    args = parser.parse_args()

    train()

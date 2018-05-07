import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from constants import *


class RNN_network:
    def __init__(self):
        self.batchX_placeholder = tf.placeholder(tf.float32, [None, None, feature_number])
        self.batchY_placeholder = tf.placeholder(tf.float32, [None, None, feature_number])

        self.net = tf.make_template('net', self._net)
        self()

    def __call__(self):
        return self.net()

    def _net(self):
        cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
        #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)

        output_rnn, self.current_state = tf.nn.dynamic_rnn(cell, self.batchX_placeholder, dtype=tf.float32)

        input_size = self.shape(self.batchX_placeholder)[2]
        output = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu)

        return output

    def loss(self):
        output = self()
        return tf.reduce_mean(tf.square(output - self.batchY_placeholder))

    def shape(self, tensor):
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from constants import *
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell


class RNN_network:
    def __init__(self):
        self.batchX_placeholder = tf.placeholder(tf.float32, [None, time_max, feature_number])
        self.batchY_placeholder = tf.placeholder(tf.float32, [None, time_max, feature_number])

        self.net = tf.make_template('net', self._net)
        self()

    def __call__(self):
        return self.net()

    def _net(self):
        cell = MultiRNNCell([GRUCell(hidden_size) for _ in range(n_layer)])

        output_rnn, self.current_state = tf.nn.dynamic_rnn(cell, self.batchX_placeholder, dtype=tf.float32)

        input_size = self.shape(self.batchX_placeholder)[2]
        # Dense Layer for the dropout
        #dense = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu)
        #dropout = tf.layers.dropout(inputs = dense, rate = dropout_rate)

        #Final layer
        output = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu)

        return output

    def loss(self):
        output = self()
        return tf.reduce_mean(tf.square(output - self.batchY_placeholder))

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

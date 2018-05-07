import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from constants import *


class RNN_network:
    def __init__(self):
        self.batchX_placeholder = tf.placeholder(tf.float32, [None, None, feature_number])
        self.batchY_placeholder = tf.placeholder(tf.float32, [None, None, feature_number])

        # Forward pass
        cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
        #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)

        #self.initial_state = cell.zero_state(batch_size, tf.float32)

        output_rnn, self.current_state = tf.nn.dynamic_rnn(cell, self.batchX_placeholder, dtype=tf.float32)

        input_size = self.shape(self.batchX_placeholder)[2]

        self.output = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu)

        self.total_loss = tf.reduce_mean(tf.square(self.output - self.batchY_placeholder))

        self.train_step = tf.train.AdagradOptimizer(0.01).minimize(self.total_loss)

    def fit(self, generateData):

        print("Start training")

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            plt.ion()
            plt.figure()
            plt.show()
            loss_list = []

            for epoch_idx in range(num_epochs):
                x,y = generateData()

                print("New data, epoch", epoch_idx)

                batchX = x
                batchY = y

                _total_loss, _train_step, _output = sess.run(
                    [self.total_loss, self.train_step, self.output],
                    feed_dict={
                        self.batchX_placeholder:batchX,
                        self.batchY_placeholder:batchY
                    })

                loss_list.append(_total_loss)

                print("Loss", _total_loss)
                self.plot(loss_list, batchX, batchY)
            print(batchY[0][0])
            print(_output[0][0])

        plt.ioff()
        plt.show()


    def plot(self, loss_list, batchX, batchY):
        plt.cla()
        plt.plot(loss_list)
        plt.draw()
        plt.pause(0.0001)

    def shape(self, tensor):
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])

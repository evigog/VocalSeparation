import network
from constants import *
import numpy as np
import data
import tensorflow as tf
import argparse
import os
import shutil

i = 0

def getBatch1(X, Y):
    global i
    X_batch = X[i]
    Y_batch = Y[i]
    i += 1

    return [X_batch], [Y_batch]

def getBatch(X, Y):
    X, Y = self.unison_shuffle(X.T, Y.T)
    return X.T, Y.T

def train():
    X, Y = data.load_batch()

    net = network.RNN_network()
    total_loss = net.loss()
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    print("Start training")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_list = []

        net.load_state(sess, CKPT_PATH)

        for epoch_idx in range(num_epochs):
            print("ok")
            batchX, batchY = getBatch1(X, Y)

            print("New data, epoch", epoch_idx)

            for i in range(len(batchX)):
                _total_loss, _train_step, _output = sess.run(
                    [total_loss, optimizer, net()],
                    feed_dict={
                        net.batchX_placeholder:batchX[i],
                        net.batchY_placeholder:batchY[i]
                    })

                loss_list.append(_total_loss)

                print("Loss", _total_loss)

            if epoch_idx % 10 == 0:
                tf.train.Saver().save(sess, CKPT_PATH + '/checkpoint', global_step=i)

def setup_path(resume):
    if resume == 0:
        if os.path.exists(CKPT_PATH):
            shutil.rmtree(CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        print("New run")
        os.makedirs(CKPT_PATH)

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    parser.add_argument('resume', help = "int, 1 if you want to continue a previous training else 0.", type = int)

    args = parser.parse_args()

    setup_path(args.resume)
    train()

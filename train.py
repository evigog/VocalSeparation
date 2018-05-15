import network
from constants import *
import numpy as np
import data
import tensorflow as tf
import argparse
import os
import shutil

def getBatch(X, Y):
    X, Y = unison_shuffle(X.T, Y.T)
    return X.T, Y.T

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train():
    X, Y = data.load_batch()

    net = network.RNN_network()
    total_loss = net.loss()

    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_list = []

        net.load_state(sess, CKPT_PATH)

        for epoch_idx in range(num_epochs):
            batchX, batchY = getBatch(X, Y)

            print("Epoch", epoch_idx)

            for i in range(len(batchX)):
                _total_loss, _train_step, _output = sess.run(
                    [total_loss, optimizer, net()],
                    feed_dict={
                        net.batchX_placeholder:batchX[i],
                        net.batchY_placeholder:batchY[i]
                    })

                loss_list.append(_total_loss)

                print("Loss", _total_loss)

                if _total_loss == np.inf:
                    print(batchY[i])
                    exit()

            if epoch_idx % 10 == 0:
                tf.train.Saver().save(sess, CKPT_PATH + '/checkpoint', global_step=i)

def setup_path(resume):
    if resume == 0:
        if os.path.exists(CKPT_PATH):
            shutil.rmtree(CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        print("Start run from scratch")
        os.makedirs(CKPT_PATH)

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    parser.add_argument('--resume', default=0, help = "int, 1 if you want to continue a previous training else 0.", type = int)

    args = parser.parse_args()

    setup_path(args.resume)
    train()

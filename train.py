import network
from constants import *
import numpy as np
import data
import tensorflow as tf
import argparse
import os
import shutil
import time

def train(verbose):
    X, Y = data.load_batch()

    net = network.RNN_network()
    total_loss = net.loss()

    losses = []

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.Session() as sess:
        t0 = time.time()
        sess.run(tf.global_variables_initializer())

        net.load_state(sess, CKPT_PATH)

        n_batch = len(X)

        idx = list(range(n_batch))

        for epoch_idx in range(num_epochs):

            np.random.shuffle(idx)

            loss_epoch = 0

            for i in range(n_batch):
                _total_loss, _train_step, _output = sess.run(
                    [total_loss, optimizer, net()],
                    feed_dict={
                        net.batchX_placeholder:X[idx[i]],
                        net.batchY_placeholder:Y[idx[i]]
                    })
                loss_epoch += _total_loss
                if verbose == 1:
                    print("batch_loss:", _total_loss)

            # if epoch_idx % 5 == 0:
            #     tf.train.Saver().save(sess, CKPT_PATH, global_step=epoch_idx)

            t1 = time.time()
            print("epoch: " + repr(epoch_idx) + " || loss_epoch: " + repr(loss_epoch) + " ||", end=' ')
            timer(t0, t1)
            losses.append(loss_epoch)
        tf.train.Saver().save(sess, SAVE_PATH + "/" + repr(time.time()) + "/" + "save.ckpt")


    print("finished.")
    losses = np.array(losses)

    np.save(SAVE_PATH + "losses.npy", losses)

def setup_path(resume):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if resume == 0:
        if os.path.exists(CKPT_PATH):
            shutil.rmtree(CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        print("Start run from scratch")
        os.makedirs(CKPT_PATH)

def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("timer: {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    parser.add_argument('--resume', default=0, help = "int, 1 if you want to continue a previous training else 0.", type = int)
    parser.add_argument('--verbose', default=0, help = "int, 1 if you want the batch loss else 0.", type = int)

    args = parser.parse_args()

    setup_path(args.resume)
    train(args.verbose)

import network
from constants import *
import numpy as np
import data
import tensorflow as tf
import argparse
import os
import shutil
import time
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib

def train(verbose):

    X_train, Y_train, X_dev, Y_dev = data.load_batch()  #load training and validation batches

    net = network.RNN_network()
    total_loss = net.loss()

    training_losses = []
    validation_losses = []

    #adaptive learning rate
    global_step = tf.Variable(0, trainable=False)
    # adaptive_learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 1400, learning_decay, staircase=True)
    optimizer_a = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer_a.minimize(total_loss, global_step=global_step)

    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    with tf.Session() as sess:

        t0 = time.time()
        sess.run(tf.global_variables_initializer(), options=run_options)

        net.load_state(sess, CKPT_PATH)

        n_train_batch = len(X_train)

        print("Number of training batch:", n_train_batch)

        idx_train = list(range(n_train_batch))

        n_dev_batch = len(X_dev)

        print("Number of validation batch:", n_dev_batch)

        idx_dev = list(range(n_dev_batch))

        for epoch_idx in range(num_epochs):
            np.random.shuffle(idx_train)

            loss_epoch = 0
            #training mode
            for i in range(n_train_batch):
                _total_loss, _train_step = sess.run(
                    [total_loss, optimizer],
                    feed_dict={
                        net.batchX_placeholder:X_train[idx_train[i]],
                        net.batchY_placeholder:Y_train[idx_train[i]]
                    })
                loss_epoch += _total_loss
                if verbose == 1:
                    print("batch_loss:", _total_loss)

            t1 = time.time()
            print("\nepoch: " + repr(epoch_idx) + " || loss_epoch: " + repr(loss_epoch) + " || ", end=' ')

            timer(t0, t1)
            training_losses.append(loss_epoch)

            if epoch_idx % 10 == 0:
                 tf.train.Saver().save(sess, CKPT_PATH, global_step=epoch_idx)

            if epoch_idx % 2 == 0: #validation mode

                dev_loss_epoch = 0
                for j in range(n_dev_batch):
                    _total_dev_loss = sess.run(
                        [total_loss],
                        feed_dict={
                            net.batchX_placeholder: X_dev[idx_dev[j]],
                            net.batchY_placeholder: Y_dev[idx_dev[j]]
                        })
                dev_loss_epoch += _total_dev_loss   #dev loss across all validation batches
                print('********epoch: '+ repr(epoch_idx) + " || validation loss: " + repr(dev_loss_epoch) + " || ", end=' ')
                validation_losses.append(dev_loss_epoch)


        tf.train.Saver().save(sess, SAVE_PATH + "/" + repr(time.time()) + "/" + "save.ckpt")

    print("finished.")
    training_losses = np.array(training_losses)
    np.save(SAVE_PATH + "training_losses.npy", training_losses)
    validation__losses = np.array(validation_losses)
    np.save(SAVE_PATH + "validation_losses.npy",  validation__losses)

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
    print(device_lib.list_local_devices())
    setup_path(args.resume)
    train(args.verbose)

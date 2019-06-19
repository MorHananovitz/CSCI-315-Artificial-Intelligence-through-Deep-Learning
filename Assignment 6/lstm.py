from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class LSTM:
    def __init__(self, n_input, n_hidden, n_output, learning_rate):
        # Target log path
        logs_path = '/tmp/tensorflow/rnn_words'
        self.writer = tf.summary.FileWriter(logs_path)

        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input, 1])
        self.y = tf.placeholder("float", [None, n_output])

        # RNN output node weights and biases
        weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_output]))}
        biases = {'out': tf.Variable(tf.random_normal([n_output]))}

        # reshape to [1, n_input]
        x = tf.reshape(self.x, [-1, n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, n_input, 1)

        # 1-layer LSTM with n_hidden units but with lower accuracy.
        # Average Accuracy= 90.60% 50k iter
        # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
        rnn_cell = rnn.BasicLSTMCell(n_hidden)

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        self.pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

        # Loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Model evaluation
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

    def run(self):
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.writer.add_graph(self.sess.graph)
        return self.sess

    def teststep(self, inputs):
        onehot_pred = self.sess.run(self.pred, feed_dict={self.x: inputs})
        return int(tf.argmax(onehot_pred, 1).eval())

    def trainstep(self, inputs, targets):
        _, acc, loss = self.sess.run([self.optimizer, self.accuracy, self.cost], feed_dict = {self.x: inputs, self.y: targets})
        return acc, loss
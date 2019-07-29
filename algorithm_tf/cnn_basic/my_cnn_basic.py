# -*- coding: utf-8 -*-
import tensorflow as tf
from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data

##########################
### SETTINGS
##########################

## Hyperparameters
dropout_keep_proba=0.5
learning_rate = 0.1
batch_size = 16
epochs = 3

# Architecture
input_size = 784
image_width = 28
image_height= 28
n_classes = 10

# Other
print_interval = 500
random_seed = 123

##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", one_hot=True)

##########################
### WRAPPER FUNCTIONS
##########################

def conv2d(input_tensor, output_channel, filter_size=(5, 5), 
           strides=[1, 1, 1, 1], padding="SAME",
           activation=None, seed=None, name="conv2d"):

    with tf.name_scope(name):
        input_channel = input_tensor.get_shape().as_list()[-1]
        weight_shape = (filter_size[0], filter_size[1],
                        input_channel, output_channel)
        weights = tf.Variable(tf.truncated_normal(shape=weight_shape,
                                     mean=0,
                                     stddev=0.01,
                                     dtype=tf.float32,
                                     seed=seed),
                             name="weights")

        biases = tf.Variable(tf.zeros(shape=(output_channel,)), name="biases")
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding)

        output = conv + biases
        if activation:
            output = activation(output)

        return output


def fully_connected(input_tensor, output_node,
                    activation=None, seed=None,
                    name="fully_connected"):
    with tf.name_scope(name):
        print(input_tensor)
        input_node = input_tensor.get_shape().as_list()[1]
        weights_shape = (input_node, output_node)
        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0,
                                                  stddev=0.01,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name="weights")
        biases = tf.Variable(tf.zeros([output_node]), name="biases")

        act = tf.matmul(input_tensor, weights) + biases
        if activation:
            act = activation(act)

        return act

##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():

    tf.set_random_seed(random_seed)
    # Input data
    X = tf.placeholder(tf.float32, [None, input_size, 1], name="inputs")
    y = tf.placeholder(tf.float32, [None, n_classes], name="target")
    keep_proba = tf.placeholder(tf.float32, None, name="keep_proba")

    input_tensor = tf.reshape(X, shape=[-1, image_height, image_width, 1])

    conv1 = conv2d(input_tensor=input_tensor,
                   output_channel=8,
                   filter_size=(3, 3),
                   strides=[1, 1, 1, 1],
                   activation=tf.nn.relu,
                   name="conv1")

    pool1 = tf.nn.max_pool(conv1,
                           ksize=(1, 2, 2, 1),
                           strides=(1, 1, 1, 1),
                           padding="SAME",
                           name="maxpool1")

    conv2 = conv2d(input_tensor=pool1,
                   output_channel=16,
                   filter_size=(3, 3),
                   strides=(1, 1, 1, 1),
                   activation=tf.nn.relu,
                   name="conv2")

    pool2 = tf.nn.max_pool(conv2,
                           ksize=(1, 2, 2, 1),
                           strides=(1, 1, 1, 1),
                           padding="SAME",
                           name="maxpool2")

    dims = pool2.get_shape().as_list()[1:]
    dims = reduce(lambda x,y : x * y, dims, 1)
    flat = tf.reshape(pool2, shape=(-1, dims))
    fc = fully_connected(flat,
                         output_node=64,
                         activation=tf.nn.relu)
    fc = tf.nn.dropout(fc, keep_prob=keep_proba)
    output_layers = fully_connected(fc,
                                   n_classes,
                                   activation=None,
                                   name="logits")
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=output_layers,
                                                   labels=y)
    cost = tf.reduce_mean(loss, name="cost")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name="train")

    # Prediction
    # TODO not arg_max
    correct_prediction = tf.equal(tf.arg_max(y, 1),
                                  tf.arg_max(output_layers, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32),
                              name="accuracy")


import numpy as np                   
##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    np.random.seed(random_seed)

    for epoch in range(1, epochs+1):

        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size
        for i in range(total_batch):
            train_X, train_y = mnist.train.next_batch(batch_size)
            train_X = train_X[:, :, None]
            _, c = sess.run(["train", "cost:0"],
                            feed_dict={"inputs:0": train_X,
                                       "target:0": train_y,
                                       "keep_proba:0": dropout_keep_proba})

            avg_cost += c
            if not i % print_interval:
                print("Minibatch: %03d | Cost: %.3f" % (i + 1, c))
#            accuracy = sess.run("accuracy",
#                                feed_dict={"inputs:0": mnist.train.images[:, :, None],
#                                           "target:0": mnist.train.labels[:, None],
#                                           "keep_proba": 1.0}
        valid_acc = sess.run("accuracy:0",
                             feed_dict={"inputs:0": mnist.validation.images[:, :, None],
                                        "target:0": mnist.validation.labels,
                                        "keep_proba:0": 1.0})
        print("Epoch: %03d | AvgCost: %.3f" % (epoch, avg_cost / (i + 1)), end="")
#         print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
        print(" | Train/Valid ACC: %.3f" % (valid_acc))
    test_acc = sess.run("accuracy:0",
                         feed_dict={"inputs:0": mnist.test.images[:, :, None],
                                    "target:0": mnist.test.labels,
                                    "keep_proba:0": 1.0})
    print("Test Acc: %.3f" % test_acc)



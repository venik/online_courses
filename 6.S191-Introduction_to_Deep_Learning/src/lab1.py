#!/bin/env python

# 6.S191: Introduction to Deep Learning
# Lab1, part 1
# https://github.com/aamini/introtodeeplearning_labs/blob/master/lab1/Lab1_Part1.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###################################
# 1.1 - The Computation Graph
###################################
# # tensors
# a = tf.constant(15, name='a')
# b = tf.constant(61, name='b')
#
# # math operation
# c = tf.add(a, b, name='c')
# print(c)

###################################
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# c = tf.add(a, b, name='c')
# d = tf.subtract(b, 1.0, name='d')
# e = tf.multiply(c, d, name='e')
#
# with tf.Session() as session:
#     a_data, b_data = 2.0, 4.0
#     feed_dict = { a: a_data, b: b_data }
#     output = session.run([e], feed_dict=feed_dict)
#     print(output)

###################################
# 1.2 - Neural Networks in TensorFlow
###################################
# n_input_nodes = 2
# n_output_nodes = 1
# x = tf.placeholder(tf.float32, (None, n_input_nodes), name='x')
# W = tf.Variable(tf.ones((n_input_nodes, n_output_nodes)), dtype=tf.float32, name='W')
# b = tf.Variable(tf.zeros(n_output_nodes), dtype=tf.float32, name='b')
#
# # math operation
# z = tf.matmul(x, W) + b
# out = tf.sigmoid(z)
#
# test_input = [[.25, .15]]
# graph = tf.Graph()
# with tf.Session() as session:
#     tf.global_variables_initializer().run(session=session)
#     feed_dict = {x: test_input}
#     output = session.run([out], feed_dict=feed_dict)
#     print(output[0])

###################################
# 1.3 - Eager execution
###################################
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
#
#
# def f(x):
#     # f(x) = x^2 + 3
#     return tf.multiply(x, x) + 3
#
#
# print('f(4) = %.2f' % f(4.))
#
# # first order derivative
# df = tfe.gradients_function(f)
# print('df(4) = %.2f' % df(4.)[0])
#
# # second order derivative
# d2f = tfe.gradients_function(df)
# print('d2f(4) = %.2f' % d2f(4.0)[0])

###################################
# 1.3.3 - Dynamic Models - Collatz conjecture
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
a = tf.constant(12, dtype=tf.float32)
counter = 0
while not tf.equal(a, 1):
    if tf.equal(a % 2, 0):
        a /= 2
    else:
        a = 3 * a + 1
    print(a)

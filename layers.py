import tensorflow as tf
import numpy as np


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name) as scope:
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def instance_norm(x):

    with tf.variable_scope("instance_norm") as scope:
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out


def linear1d(inputlin, inputdim, outputdim, name="linear1d", std=0.02, mn=0.0):

    with tf.variable_scope(name) as scope:

        weight = tf.get_variable("weight",[inputdim, outputdim])
        bias = tf.get_variable("bias",[outputdim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

        return tf.matmul(inputlin, weight) + bias


def general_conv2d(inputconv, o_d=64, f_h=5, f_w=5, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, norm_type='batch_norm', do_relu=True, relufactor=0):
    with tf.variable_scope(name) as scope:

        conv = tf.contrib.layers.conv2d(inputconv, o_d, [f_w, f_h], [s_w, s_h], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            if norm_type == 'instance':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv



def general_deconv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, norm_type='batch_norm', do_relu=True, relufactor=0):
    with tf.variable_scope(name) as scope:

        conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            if norm_type == 'instance':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

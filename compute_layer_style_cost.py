import tensorflow as tf
from gram_matrix import gram_matrix
import numpy as np


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_S.get_shape().as_list()
    a_S_unrolled = tf.reshape(tf.transpose(a_S), [n_C, n_H * n_W])
    a_G_unrolled = tf.reshape(tf.transpose(a_G), [n_C, n_H * n_W])
    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)

    J_style_layer = tf.reduce_sum((tf.square(tf.subtract(GS, GG)))) / (4 * np.square(n_C) * np.square(n_H * n_W))

    return J_style_layer

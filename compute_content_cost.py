import tensorflow as tf


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(tf.transpose(a_C), [m, n_H * n_W * n_C])
    a_G_unrolled = tf.reshape(tf.transpose(a_G), [m, n_H * n_W * n_C])

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content

import tensorflow as tf


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))

    return GA

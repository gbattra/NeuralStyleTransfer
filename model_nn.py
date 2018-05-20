import tensorflow as tf
import matplotlib.pyplot as plt


def model_nn(sess, model, train_step, J, num_iterations = 200):
    sess.run(tf.global_variables_initializer())

    for i in range(num_iterations):
        _, loss = sess.run([train_step, J])
        print(loss)

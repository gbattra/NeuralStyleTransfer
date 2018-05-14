import tensorflow as tf
import matplotlib.pyplot as plt


def model_nn(sess, model, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    generated_image = sess.run(model['conv4_2'])

    plt.imshow(generated_image)

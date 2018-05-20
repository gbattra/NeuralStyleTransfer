import tensorflow as tf
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import h5py
from utils import reshape_and_normalize_image, generate_noise_image
from compute_content_cost import compute_content_cost
from compute_layer_style_cost import compute_layer_style_cost
from compute_total_style_cost import compute_total_style_cost
from total_cost import total_cost
from gram_matrix import gram_matrix
from model_nn import model_nn

content_image = scipy.misc.imread('images/content/cat_2.jpg')
# plt.imshow(content_image)
# plt.show()

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print(J_content.eval())  # should be 6.76559

style_image = scipy.misc.imread('images/style/style_2.jpg')
# plt.imshow(style_image)
# plt.show()

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    print(GA.eval())  # [0][0] = 6.4223...

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    print(J_style_layer.eval())  # 9.19028

STYLE_LAYERS = [
    (1, 0.2),
    (2, 0.2),
    (3, 0.2),
    (4, 0.2),
    (5, 0.2)]

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print(J)  # 35.34667...

tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = reshape_and_normalize_image(content_image)
style_image = reshape_and_normalize_image(style_image)
generated_image = generate_noise_image(content_image)

plt.imshow(generated_image[0])
# plt.show()

vgg = tf.keras.applications.VGG19(input_tensor=tf.convert_to_tensor(content_image, dtype='float32'))
out = vgg.layers[4].output
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

vgg = tf.keras.applications.VGG19(input_tensor=tf.convert_to_tensor(style_image, dtype='float32'))
J_style = compute_total_style_cost(vgg, STYLE_LAYERS, sess)

J = total_cost(J_content, J_style)

optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)

vgg = tf.keras.applications.VGG19(input_tensor=tf.convert_to_tensor(generated_image, dtype='float32'))
model_nn(sess, vgg, train_step, J)

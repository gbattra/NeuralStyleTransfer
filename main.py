import tensorflow as tf
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import h5py

vgg = tf.keras.applications.VGG19()
print(vgg)

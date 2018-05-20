import numpy as np


class CONFIG:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    OUTPUT_DIR = 'output/'

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # crop image
    image = image[:CONFIG.IMAGE_HEIGHT, :CONFIG.IMAGE_WIDTH]

    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))

    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS

    return image

def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image

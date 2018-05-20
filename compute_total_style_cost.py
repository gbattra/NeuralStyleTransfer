from compute_layer_style_cost import compute_layer_style_cost
import tensorflow as tf


def compute_total_style_cost(model, STYLE_LAYERS, sess):
    J_style = 0

    for layer_index, coeff in STYLE_LAYERS:
        out = model.layers[layer_index].output
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(tf.convert_to_tensor(a_S, dtype='float32'), a_G)

        J_style += coeff * J_style_layer

    return J_style

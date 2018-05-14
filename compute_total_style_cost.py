from compute_layer_style_cost import compute_layer_style_cost


def compute_total_style_cost(model, STYLE_LAYERS, sess):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style

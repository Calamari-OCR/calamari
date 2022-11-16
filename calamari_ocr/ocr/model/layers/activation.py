import tensorflow.keras as keras


def activation_by_str(a: str):
    if a is None:
        return
    elif a == "leaky_relu":
        return leaky_relu()
    else:
        return getattr(keras.activations, a)


def leaky_relu(leak=0.1, name="LeakyRelu"):
    return keras.layers.LeakyReLU(alpha=leak, name=name)

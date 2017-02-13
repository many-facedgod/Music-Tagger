from Totem import *


def get_model(input_shape, top, rand):
    """
    Returns a generated model
    :param input_shape: The shape of the input that the model takes
    :param top: The number of outputs
    :param rand: An RNG object
    :return: The generated model
    """
    m = model.Model(input_shape)
    m.add_layer(layers.ConvLayer("Conv1", 100, (3, 4), rand, activation="relu", init_method="glorot"))
    m.add_layer(layers.PoolLayer("Pool1", down_sample_size=(2, 3)))
    m.add_layer(layers.ConvLayer("Conv2", 100, (3, 4), rand, activation="relu", init_method="glorot"))
    m.add_layer(layers.PoolLayer("Pool2", down_sample_size=(2, 3)))
    m.add_layer(layers.BNLayer("BN1", mode="low_mem"))
    m.add_layer(layers.ConvLayer("Conv3", 150, (3, 4), rand, activation="relu", init_method="glorot"))
    m.add_layer(layers.ConvLayer("Conv4", 200, (3, 4), rand, activation="relu", strides=(2, 3), init_method="glorot"))
    m.add_layer(layers.ConvLayer("Conv5", 250, (3, 4), rand, strides=(2, 3), activation="relu", init_method="glorot"))
    m.add_layer(layers.PoolLayer("Pool3", down_sample_size=(2, 3)))
    m.add_layer(layers.FlattenLayer("Flat1"))
    m.add_layer(layers.FCLayer("FC1", n_units=1000, rng=rand, init_method="glorot"))
    m.add_layer(layers.FCLayer("FC2", top, rand, activation="sigmoid", init_method="glorot"))
    return m

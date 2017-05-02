from Totem import *

def get_model(input_shape, top, rand):
    """
    Returns a generated model
    :param input_shape: The shape of the input that the model takes
    :param top: The number of outputs
    :param rand: An RNG object
    :return: The generated model
    """
    m=model.Model(input_shape)
    m.add_layer(layers.ConvLayer("Conv1.1", 50, (3, 7), rand, mode="half"))
    m.add_layer(layers.PoolLayer("Pool1.1", (2, 4)))
    m.add_layer(layers.ConvLayer("Conv1.2", 100, (3, 5), rand, mode="half"))
    m.add_layer(layers.PoolLayer("Pool1.2", (2, 4)))
    m.add_layer(layers.ConvLayer("Conv1.3", 70, (3, 3), rand, mode="half"))
    m.add_layer(layers.PoolLayer("Pool1.3", (2, 2)))
    m.add_layer(layers.PoolLayer("SubSample1", (2, 4), mode="avg"), source="inputs")
    m.add_layer(layers.ConvLayer("Conv2.1", 100, (3, 5), rand, mode="half"))
    m.add_layer(layers.PoolLayer("Pool2.1", (2, 4)))
    m.add_layer(layers.ConvLayer("Conv2.2", 70, (3, 3), rand, mode="half"))
    m.add_layer(layers.PoolLayer("Pool2.2", (2, 2)))
    m.add_layer(layers.PoolLayer("SubSample2", (2, 4), mode="avg"), source="SubSample1")
    m.add_layer(layers.ConvLayer("Conv3.1", 70, (3, 3), rand, mode="half"))
    m.add_layer(layers.PoolLayer("Pool3.1", (2, 2)))
    m.add_layer(layers.JoinLayer("Join1", axis=1), source=("Pool1.3", "Pool2.2", "Pool3.1"))
    m.add_layer(layers.ConvLayer("ConvLast1", 70, (3, 3), rand, mode="half"))
    m.add_layer(layers.ConvLayer("ConvLast2", 70, (3, 3), rand, mode="half"))
    m.add_layer(layers.PoolLayer("PoolLast", (2, 2)))
    m.add_layer(layers.FlattenLayer("FlattenLast"))
    m.add_layer(layers.BNLayer("BNLast"))
    m.add_layer(layers.DropOutLayer("Dropout1", rand, 0.6))
    m.add_layer(layers.FCLayer("FCLast", 500, rand))
    m.add_layer(layers.FCLayer("Top", top, rand, activation="sigmoid"))
    return m
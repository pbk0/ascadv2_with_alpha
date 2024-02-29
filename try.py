import tensorflow as tf
from tensorflow import keras as ke
import numpy as np

_in = np.zeros((100,33,2))

_in_layer = ke.Input(shape=(33,1))

_conv_layer = ke.layers.Conv1D(kernel_size=(3,), strides=1, filters=3, activation='selu', padding='same')(_in_layer)

_model = ke.Model(inputs=_in_layer, outputs=_conv_layer)

_model(_in)
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.layers import UpSampling2D, Flatten
from tensorflow.keras.layers import Lambda


def combine_model(gen, disc, latent_dim):
    noise = Input(shape=latent_dim)
    img = gen(noise)
    disc.trainable = False
    valid = disc(img)
    return Model(noise, valid)

############################################################
# THIS FILE HAS BEEN COPIED FROM THE vwrs/dcgan-mnist REPO #
# then some modifications were made                        #         
############################################################

# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout


def build_generator2(latent_dim, units=1024,activation='relu'):
    model = Sequential()
    model.add(Dense(input_dim=latent_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


def build_discriminator2(img_shape, opt, loss='binary_crossentropy', nb_filter=64):
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same',
                     input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*nb_filter, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    disc = Model(img, validity)

    disc.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy']
    )

    return disc

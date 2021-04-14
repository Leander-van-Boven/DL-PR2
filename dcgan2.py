############################################################
# THESE ARCHITECTURES HAVE BEEN COPIED FROM THE FILE:      #
# https://github.com/vwrs/dcgan-mnist/blob/master/model.py #
############################################################

from tensorflow.keras.layers import (ELU, Activation, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, Reshape, UpSampling2D)
from tensorflow.keras.models import Model, Sequential


def build_generator2(latent_dim, units=1024, activation='relu'):
    """Builds the DCGAN generator according to the Hideaki Kanehara
        implementation.

    Parameters
    ----------
    latent_dim : tuple
        The size of the input vector, i.e. the noise vector.

    Returns
    -------
    Model
        The generator as Model (not compiled yet).
    """
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


def build_discriminator2(img_shape, include_dense=True,
                         compile=True, opt='adam', nb_filter=64):
    """Builds the DCGAN discriminator accoring to the Hideaki Kanehara
    implementation.

    Parameters
    ----------
    img_shape : tuple
        The input dimension of the discriminator.
    include_dense : bool, optional
        Whether to add the boolean dense layer to the end of the model,
        by default True
    compile : bool, optional
        Whether to compile the discriminator model, by default True
    opt : str/keras.Optimizer, optional
        The optimiser to use when compiling the model, by default 'adam'
    nb_filter : int, optional
        The amount of neurons to add to the narrow-band filter layer,
        by default 64

    Returns
    -------
    Model
        The discriminator as Model.
    """
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
    if include_dense:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    disc = Model(img, validity)

    if compile:
        disc.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    return disc

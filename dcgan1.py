###########################################################################
# THESE ARCHITECTURES HAVE BEEN COPIED FROM THE FILE:                     #
# https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py #
###########################################################################

from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input, LeakyReLU,
                                     Reshape, UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model, Sequential


def build_generator1(latent_dim):
    """Builds the DCGAN generator according to the Erik Linder-Norén
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

    model.add(Dense(128*7*7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


def build_discriminator1(img_shape, include_dense=True,
                         compile=True, opt='adam'):
    """Builds the DCGAN discriminator accoring to the Erik Linder-Norén
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

    Returns
    -------
    Model
        The discriminator as Model.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    if include_dense:
        model.add(Dense(1, activation='sigmoid'))

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

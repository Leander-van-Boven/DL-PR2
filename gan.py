import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.layers import UpSampling2D, Flatten
from tensorflow.keras.layers import Lambda


def build_generator(latent_dim, img_size, force_single_channel=False):
    (img_row, img_col, channels) = img_size

    row = img_row//4
    col = img_col//4

    assert row*4 == img_row and col*4 == img_col

    if force_single_channel:
        img_chan = img_size[2]
        channels = 1

    model = Sequential()

    model.add(Dense(128 * row * col, activation="relu",
                    input_dim=latent_dim))
    model.add(Reshape((row, col, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    if force_single_channel:
        c = tf.constant([1, 1, 1, img_chan], tf.int32)
        model.add(Lambda(
            lambda a : tf.tile(a, c)
        ))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


def build_discriminator(architecture, img_shape, opt):
    cnn_disc = architecture(
        include_top=False,
        weights="imagenet", # maybe try different weights? denser seems better
        input_shape=img_shape,
        pooling=None
        # classifier_activation="softmax"
    )

    cnn_disc.trainable = False

    flattened = Flatten()(cnn_disc.output)

    bool_layer = Dense(1, activation='sigmoid')(flattened)
    disc = Model(inputs=cnn_disc.input, outputs=bool_layer)

    disc.summary()

    disc.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return disc


def combine_model(gen, disc, latent_dim):
    noise = Input(shape=latent_dim)
    img = gen(noise)
    disc.trainable = False
    valid = disc(img)
    return Model(noise, valid)


if __name__ == "__main__":
    model = build_generator(100, (28, 28, 1))
    model.summary()

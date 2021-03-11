from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D
from tensorflow.keras.layers import Dense, Input, Reshape, UpSampling2D


def build_generator(noise_size, img_size):
    (img_row, img_col, channels) = img_size

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu",
                    input_dim=noise_size))
    model.add(Reshape((7, 7, 128)))
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
    # TODO: make sure output size of the model is correct

    model.summary()

    noise = Input(shape=(noise_size,))
    img = model(noise)

    return Model(noise, img)


def build_discriminator(architecture, img_shape, opt):
    cnn_disc = architecture(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling=None,
        classifier_activation="softmax"
    )
    cnn_disc.trainable = False

    bool_layer = Dense(1, activation='sigmoid')(cnn_disc.output)
    disc = Model(inputs=cnn_disc.input, outputs=bool_layer)

    disc.compile(
        optimiser=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return disc


def combine_model(gen, disc):
    noise = Input(shape=gen.layers[0].shape)
    img = gen(noise)
    valid = disc(img)
    return Model(noise, valid)

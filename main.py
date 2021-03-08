import numpy as np

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.efficientnet\
    import EfficientNetB0, EfficientNetB7
from tensorflow.keras.datasets import mnist, cifar10

from gan import build_generator, build_discriminator
from experiment import run_experiment


if __name__ == "__main__":
    datasets = {
        "mnist": mnist,     # (28, 28, 1)
        "cifar10": cifar10  # (32, 32, 3), CIFAR100 has same shape
    }

    disc_architectures = {
        "irv2": InceptionResNetV2,
        "efnb0": EfficientNetB0,
        "efnb7": EfficientNetB7
    }

    ###########################
    # Add argparse logic here #
    ###########################
    # TODO: needs cmd args, these values are out of thin air
    noise_size = 100
    opt = 'adam'
    epochs = 500
    batch_size = 32

    (X_train, y_train), (X_test, y_test) = datasets[0].load_data()
    # Make sure (row, col) becomes (row, col, chan)
    if len(X_train.shape[1:]) == 2:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
    img_shape = X_train.shape[1:]

    architecture = disc_architectures[0]

    gen = build_generator(noise_size, img_shape)
    disc = build_discriminator(architecture, img_shape)

    run_experiment(gen, disc, X_train, opt, epochs, batch_size)

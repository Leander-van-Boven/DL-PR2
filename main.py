import argparse
import logging
import numpy as np

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.efficientnet\
    import EfficientNetB0, EfficientNetB7
from tensorflow.keras.datasets import mnist, cifar10

from gan import build_generator, build_discriminator
from experiment import run_experiment


def main(args):
    # TODO: needs cmd args, these values are out of thin air
    noise_size = tuple(args.input_size)
    opt = args.optimizer
    epochs = args.epochs
    batch_size = args.batch

    (X_train, y_train), (X_test, y_test) = args.dataset.load_data()
    # Make sure (row, col) becomes (row, col, chan)
    if len(X_train.shape[1:]) == 2:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
    img_shape = X_train.shape[1:]

    architecture = args.disc_arch

    gen = build_generator(noise_size, img_shape)
    disc = build_discriminator(architecture, img_shape, opt)

    run_experiment(gen, disc, X_train, opt, epochs, batch_size, args.log_dir)


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

    # TODO: Change string value to class constructor, add argument
    #       for optimizer parameters (*args, **kwargs)
    optimizers = {
        'adam': 'adam'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbosity', type=int, choices=[1, 2, 3, 4, 5], default=2,
        help='verbosity level. 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR, 5=CRITICAL'
    )
    parser.add_argument(
        '-d', '--dataset', type=datasets.get, choices=datasets.keys(),
        default='mnist', help='the dataset to use in the experiment'
    )
    parser.add_argument(
        '-a', '--disc_arch', type=disc_architectures.get,
        choices=disc_architectures.keys(), default='efnb0',
        help='the architecture to use for the discriminator'
    )
    parser.add_argument(
        '-o', '--optimizer', type=optimizers.get, choices=optimizers.keys(),
        default='adam', help='the optimizer to use'
    )
    parser.add_argument(
        '-s', '--latent_dim', type=int, default=100,
        help='the dimensionality of the latent space used for input noise'
    )
    parser.add_argument(
        '-b', '--batch', type=int, choices=[i*8 for i in range(1, 33)],
        default=32, help='the batch size'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=500,
        help='amount of training epochs'
    )
    parser.add_argument(
        '-l', '--log_dir', type=str, default='~/',
        help='output location for training and test logs'
    )

    args = parser.parse_args()

    # note(Ramon): this is nice, but when running on Peregrine, the entire
    # terminal output of the program is saved to a log file by default so I'm
    # not sure it's necessary
    logging.basicConfig(
        level=args.verbosity, datefmt='%I:%M:%S',
        format='[%(asctime)s] (%(levelno)s) %(message)s'
    )

    main(args)

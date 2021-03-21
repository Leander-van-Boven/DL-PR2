import argparse
import logging
import os
import numpy as np
import tensorflow as tf
from functools import partial
from numpy.core.shape_base import atleast_3d

from tensorflow.keras.applications import InceptionResNetV2, ResNet152V2
from tensorflow.keras.applications.efficientnet\
    import EfficientNetB0, EfficientNetB7
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.optimizers import Adam

from gan import build_generator, build_discriminator
from experiment import run_experiment
from dcgan_disc import build_dcgan_discriminator

from set_session import initialize_session


def main(args):
    initialize_session()

    noise_size = args.latent_dim
    opt = args.optimizer
    epochs = args.epochs
    batch_size = args.batch

    (x_train, y_train), (x_test, _) = args.dataset.load_data()

    # Make sure (row, col) becomes (row, col, chan) (mnist is grayscale)
    force_single_channel = False
    if len(x_train.shape[1:]) == 2:
        # x_train = x_train[y_train == 7, :, :]
        force_single_channel = True
        x_train = process_for_mnist(x_train)
        x_test = process_for_mnist(x_test)

    # show_training_image(x_train)

    img_shape = x_train.shape[1:]
    architecture = args.disc_arch

    gen = build_generator(noise_size, img_shape, force_single_channel)
    if architecture == 'dcgan':
        disc = build_dcgan_discriminator(img_shape, opt)
    else:
        disc = build_discriminator(architecture, img_shape, opt)

    log_interval = epochs // int(epochs * args.log_interval)

    run_experiment(
        gen, disc, x_train, opt, epochs, batch_size, noise_size, args.log_dir,
        log_interval
    )


def show_training_image(x_train):
    import matplotlib.pyplot as plt
    _, axs = plt.subplots()
    axs.imshow(
        x_train[np.random.randint(x_train.shape[0]), :, :, :].astype(np.uint8)
    )
    plt.show()
    exit()


def process_for_mnist(imgs):
    imgs = np.expand_dims(imgs, -1)
    imgs = tf.convert_to_tensor(imgs, dtype=tf.uint8)
    # InceptionResNet needs at least 75x75 + needs to be divisible by 4
    # imgs = tf.image.resize(imgs, [76, 76],
    #                        method=tf.image.ResizeMethod.LANCZOS3)
    imgs = tf.image.pad_to_bounding_box(imgs, 2, 2, 32, 32)
    imgs = tf.image.grayscale_to_rgb(imgs)
    imgs = np.array(imgs)
    return imgs


if __name__ == "__main__":
    def get_dict_val(dict, val):
        return dict[val]

    datasets = {
        "mnist": mnist,     # (28, 28, 1)
        "cifar10": cifar10  # (32, 32, 3), CIFAR100 has same shape
    }

    disc_architectures = {
        # "irv2": InceptionResNetV2,
        "r152v2": ResNet152V2,
        "efnb0": EfficientNetB0,
        "efnb7": EfficientNetB7,
        "dcgan": "dcgan"
    }

    # TODO: Change string value to class constructor, add argument
    #       for optimizer parameters (*args, **kwargs)
    optimizers = {
        'adam': Adam(0.0002, 0.5),
        'mse': 'mse'
    }

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-v', '--verbosity', type=int, choices=[1, 2, 3, 4, 5], default=2,
    #     help='verbosity level. 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR, 5=CRITICAL'
    # )
    parser.add_argument(
        '-d', '--dataset', type=partial(get_dict_val, datasets),
        default='mnist', help='the dataset to use in the experiment'
    )
    parser.add_argument(
        '-a', '--disc_arch', type=partial(get_dict_val, disc_architectures),
        default='efnb0', help='the architecture to use for the discriminator'
    )
    parser.add_argument(
        '-o', '--optimizer', type=partial(get_dict_val, optimizers),
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
        '-l', '--log_dir', type=str, default='../',
        help='output location for training and test logs'
    )
    parser.add_argument(
        '-i', '--log_interval', type=float, default=.1,
        help='fraction of epochs on which to save the current images'
    )

    args = parser.parse_args()

    # note(Ramon): this is nice, but when running on Peregrine, the entire
    # terminal output of the program is saved to a log file by default so I'm
    # not sure it's necessary
    # logging.basicConfig(
    #     level=args.verbosity, datefmt='%I:%M:%S',
    #     format='[%(asctime)s] (%(levelno)s) %(message)s'
    # )

    main(args)

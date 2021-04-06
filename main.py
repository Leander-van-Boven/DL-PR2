import argparse
import csv
import logging
import os
import datetime
import numpy as np
import tensorflow as tf
from functools import partial
from numpy.core.shape_base import atleast_3d

from tensorflow.keras.applications import InceptionResNetV2, ResNet152V2
from tensorflow.keras.applications.efficientnet\
    import EfficientNetB0, EfficientNetB7
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.optimizers import Adam, Nadam

from gan import build_generator, build_discriminator
from experiment import run_experiment
from dcgan_disc import build_dcgan_discriminator

from set_session import initialize_session


def main(args):
    if args.init_session:
        initialize_session()

    noise_size = args.latent_dim
    opt = args.optimizer
    epochs = args.epochs
    batch_size = args.batch
    architecture = args.disc_arch

    # Create output directory
    (log_path, img_path) = prepare_directory(args.log_dir)

    # Write experimental setup to file
    log_setup(log_path, args)

    # Load data set
    (x_train, _), (x_test, _) = args.dataset.load_data()

    # Make sure (row, col) becomes (row, col, chan) (mnist is grayscale)
    force_single_channel = False
    if len(x_train.shape[1:]) == 2:
        force_single_channel = True
        x_train = process_for_mnist(x_train)
        x_test = process_for_mnist(x_test)

    # show_training_image(x_train)
    noise = tf.random_normal(
        shape=tf.shape(x_train), mean=0.0, stddev=1, dtype=tf.float32
    )
    x_train = tf.add(x_train, noise)

    img_shape = x_train.shape[1:]

    gen = build_generator(noise_size, img_shape, force_single_channel)
    if architecture == 'dcgan':
        disc = build_dcgan_discriminator(img_shape, opt)
    else:
        disc = build_discriminator(architecture, img_shape, opt)

    # save an image on a fraction of the log interval
    log_interval = int(epochs * args.log_interval)

    run_experiment(
        gen, disc, x_train, opt, epochs, batch_size, noise_size, log_path, 
        img_path, log_interval
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
    # append a dimension at the end
    imgs = np.expand_dims(imgs, -1)
    # convert to tensor for image processing
    imgs = tf.convert_to_tensor(imgs, dtype=tf.uint8)
    # add padding
    imgs = tf.image.pad_to_bounding_box(imgs, 2, 2, 32, 32)
    # convert 1d to 3d
    imgs = tf.image.grayscale_to_rgb(imgs)
    # convert back to np array
    imgs = np.array(imgs)
    return imgs


def prepare_directory(log_dir):
    # prepare log output directory. name is YYYY-MM-DDTHH:MM
    run_time = datetime.datetime.now().isoformat(timespec='minutes')
    run_time = run_time.replace(':', '-')

    log_path = os.path.join(log_dir, run_time)
    img_path = os.path.join(log_path, "images")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    return log_path, img_path


def log_setup(log_path, args):
    # Print experiment setup for reference
    setup_file = os.path.join(log_path, "setup.txt")
    with open(setup_file, mode='w') as file:
        file.writelines([f"{key:20} = {value}\n" for key, value in vars(args).items()])


def float_range(mini,maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - maximum acceptable argument
         maxi - maximum acceptable argument
         
       Taken from https://stackoverflow.com/a/64259328/4545692"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError(
                "must be in range [" + str(mini) + " .. " + str(maxi)+"]"
            )
        return f

    # Return function handle to checking function
    return float_range_checker


if __name__ == "__main__":
    def get_dict_val(dict, val):
        return dict[val]

    datasets = {
        "mnist": mnist,     # (28, 28, 1)
        "cifar10": cifar10  # (32, 32, 3), CIFAR100 has same shape
    }

    disc_architectures = {
        "r152v2": ResNet152V2,
        "efnb0": EfficientNetB0,
        "efnb7": EfficientNetB7,
        "dcgan": "dcgan"
    }

    # TODO: Change string value to class constructor, add argument
    #       for optimizer parameters (*args, **kwargs)
    optimizers = {
        'adam': Adam(0.01, 0.9, 0.9), # based on https://arxiv-org.proxy-ub.rug.nl/pdf/1906.11613.pdf
        'nadam': Nadam(0.01, 0.9, 0.9),
        'mse': 'mse'
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-a', '--disc_arch', type=partial(get_dict_val, disc_architectures),
        default='efnb0', help='the architecture to use for the discriminator'
    )
    parser.add_argument(
        '-b', '--batch', type=int, choices=[i*8 for i in range(1, 33)],
        default=32, help='the batch size'
    )
    parser.add_argument(
        '-d', '--dataset', type=partial(get_dict_val, datasets),
        default='mnist', help='the dataset to use in the experiment'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=500,
        help='amount of training epochs'
    )
    parser.add_argument(
        '-g', '--init_session', type=bool, default=False,
        help='whether the program should manually set the gpu session'
    )
    parser.add_argument(
        '-i', '--log_interval', type=float_range(0, 0.5), default=.05,
        help='fraction of epochs on which to save the current images.\
              setting this to 0 will save no images.'
    )
    parser.add_argument(
        '-o', '--optimizer', type=partial(get_dict_val, optimizers),
        default='adam', help='the optimizer to use'
    )
    parser.add_argument(
        '-l', '--log_dir', type=str, default='./experiments/',
        help='output location for training and test logs'
    )
    parser.add_argument(
        '-s', '--latent_dim', type=int, default=100,
        help='the dimensionality of the latent space used for input noise'
    )

    args = parser.parse_args()

    main(args)

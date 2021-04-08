import argparse
import csv
import logging
import os
import datetime
import numpy as np
import tensorflow as tf
from functools import partial
from numpy.core.shape_base import atleast_3d

from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Nadam

from experiment import run_experiment
from dcgan1 import build_generator1
from dcgan2 import build_generator2

from set_session import initialize_session


def main(args):
    if args.init_session:
        initialize_session()

    noise_size = args.latent_dim
    opt = args.optimizer
    epochs = args.epochs
    batch_size = args.batch

    # Create output directory
    (log_path, img_path) = prepare_directory(args.log_dir)

    # Write experimental setup to file
    log_setup(log_path, args)

    exp_data = mnist if args.dataset == 'digits' else fashion_mnist
    disc_init = args.dataset if not args.notransfer else \
        'fashion' if args.dataset == 'digits' else 'digits'

    # Load data set
    (x_train, _), (_, _) = exp_data.load_data()

    # Scale x_train between 1 and -1
    x_train = (x_train / 127.5) - 1.

    # Add noise to data (if applicable)
    if args.noise > 0:
        rand_range = args.noise
        noise = 2 * rand_range * np.random.random(x_train.size) - rand_range
        x_train += noise.reshape(x_train.shape)
        x_train = 2 * \
            ((x_train - x_train.min()) / (x_train.max() - x_train.min())) - 1.

    img_shape = x_train.shape[1:]

    # Construct or load D and G models
    gen = eval('build_generator%s(noise_size)' % args.architecture)
    disc = load_model('./discriminator%s_%s' % (args.architecture, disc_init))

    # save an image on a fraction of the log interval
    log_interval = int(epochs * args.log_interval)

    run_experiment(
        gen, disc, x_train, opt, epochs, batch_size, noise_size, log_path, 
        img_path, log_interval
    )


def show_training_image(x_train, idx=None):
    import matplotlib.pyplot as plt
    _, axs = plt.subplots()
    idx = idx or np.random.randint(x_train.shape[0])
    axs.imshow(
        (x_train[idx, :, :, :] + 1) / 2
    )
    plt.show()
    return idx


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

    # TODO: Change string value to class constructor, add argument
    #       for optimizer parameters (*args, **kwargs)
    optimizers = {
        'adam': Adam(0.01, 0.9, 0.9), # based on https://arxiv-org.proxy-ub.rug.nl/pdf/1906.11613.pdf
        'nadam': Nadam(0.01, 0.9, 0.9),
        'adamdcgan': Adam(0.0002, 0.5),
        'mse': 'mse'
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-a', '--architecture', type=int, choices=[1,2], default=1,
        help='the architecture to use'
    )
    parser.add_argument(
        '-b', '--batch', type=int, choices=[i*8 for i in range(1, 33)],
        default=32, help='the batch size'
    )
    parser.add_argument(
        '-d', '--dataset', type=str, choices=['digits', 'fashion'],
        default='digits', help='the dataset to use in the experiment'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=500,
        help='amount of training epochs'
    )
    parser.add_argument(
        '-g', '--init_session', action='store_true',
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
    parser.add_argument(
        '-n', '--noise', type=float_range(0, 1), default=0,
        help='the amount of noise to add to the trainig data set'
    )
    parser.add_argument(
        '-t', '--notransfer', action='store_true',
        help='add flag to disable transfer learning'
    )

    args = parser.parse_args()
    #input(args)

    main(args)

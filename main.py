import argparse
import datetime
import os

import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

from dcgan1 import build_discriminator1, build_generator1
from dcgan2 import build_discriminator2, build_generator2
from experiment import run_experiment
from set_session import initialize_session


def nothing(*args, **kwargs):
    pass


def main(args):
    """Main method that initializes everything that is needed
        to run the experiments.

    Parameters
    ----------
    args : ArgParse namespace
        ArgParse namespace that contains all the commandline arguments used.
    """
    if args.init_session:
        initialize_session()

    noise_size = args.latent_dim
    epochs = args.epochs
    batch_size = args.batch

    # Create output directory
    (log_path, img_path) = prepare_directory(args.log_dir)

    # Write experimental setup to file
    log_setup(log_path, args)

    # Initialize model optimizers
    pre_optimizers = {
            # based on https://arxiv-org.proxy-ub.rug.nl/pdf/1906.11613.pdf
            'adamm2m': Adam(0.001, 0.9, 0.9),
            'adamdcgan': Adam(0.005, 0.5),
        }
    try:
        opt = pre_optimizers[args.optimizer]
    except:
        optimizers = {
            'adam': Adam,
            'nadam': Nadam,
            'rmsprop': RMSprop
        }
        opt = optimizers[args.optimizer](*args.optargs)

    # Determine which dataset to use
    exp_data = mnist if args.dataset == 'digits' else fashion_mnist
    if args.dtl == 1:
        disc_init = args.dataset
    elif args.dtl == 2:
        disc_init = 'fashion' if args.dataset == 'digits' else 'digits'

    # Load data set
    (X_train, _), (_, _) = exp_data.load_data()

    # Scale X_train between 1 and -1
    X_train = (X_train / 127.5) - 1.

    # Add noise to data (if applicable)
    if args.noise > 0:
        rand_range = args.noise
        noise = 2 * rand_range * np.random.random(X_train.size) - rand_range
        X_train += noise.reshape(X_train.shape)
        X_train = 2 * \
            ((X_train - X_train.min()) / (X_train.max() - X_train.min())) - 1.

    # Construct or load D and G models
    gen = eval('build_generator%s(noise_size)' % args.architecture)

    if args.dtl == 0:
        disc = eval(
            'build_discriminator%s((28,28,1), opt=opt)' % args.architecture
        )
    else:
        disc = load_model(
            './discriminator%s_%s' % (args.architecture, disc_init)
        )
        disc.layers[1].trainable = False
        disc.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    print("DISCRIMINATOR")
    disc.summary()

    # Save an image on a fraction of the log interval
    log_interval = int(epochs * args.log_interval)

    # Determine printing method based on verbosity
    pr = nothing if args.verbose == 0 else print

    # Perform the actual experiment
    run_experiment(
        gen, disc, X_train, opt, epochs, batch_size, noise_size, log_path,
        img_path, log_interval, pr
    )


def prepare_directory(log_dir):
    """Initializes the log directory, using the current date and time.
        The format of the logging directory will look like: YYYY-MM-DDTHH:MM.

    Parameters
    ----------
    log_dir : str
        Directory in which to create the logging directory.

    Returns
    -------
    tuple
        Tuple containing the name of the logging directory
        and the subdirectory in which to save the generated images.
    """
    run_time = datetime.datetime.now().isoformat(timespec='minutes')
    run_time = run_time.replace(':', '-')

    log_path = os.path.join(log_dir, run_time)
    img_path = os.path.join(log_path, "images")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    return log_path, img_path


def log_setup(log_path, args):
    """Logs the arguments used to run the experiments with to a file.
        This way the experiment can be easily reproduced.

    Parameters
    ----------
    log_path : str
        Directory in which to store the setup details.
    args : dict
        Dictionary containing the arguments used to run the experiments.
    """
    setup_file = os.path.join(log_path, "setup.txt")
    with open(setup_file, mode='w') as file:
        file.writelines(
            [f"{key:20} = {value}\n" for key, value in vars(args).items()]
        )


def float_range(mini, maxi):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi.

    Parameters
    ----------
    mini : number
        Minimum acceptable argument.
    maxi : number
        Maximum acceptable argument.

    Returns
    -------
    Function
        Function handle to checking function.

    Raises
    ------
    argparse.ArgumentTypeError
        If the entered number is not a floating point.
    argparse.ArgumentTypeError
        If the entered floating point is not within the correct range.
    """
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

    return float_range_checker


if __name__ == "__main__":
    def get_dict_val(dict, val):
        return dict[val]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-v', '--verbose', type=int, choices=[0, 1], default=1,
        help='verbosity mode, 0 is no printing'
    )
    parser.add_argument(
        '-a', '--architecture', type=int, choices=[1, 2], default=1,
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
        help='fraction of epochs on which to save the current images, ' +
             'setting this to 0 will save no images'
    )
    parser.add_argument(
        '-o', '--optimizer', type=str,
        choices=['adamm2m', 'adamdcgan', 'adam', 'nadam', 'rmsprop'],
        default='adamdcgan', help='the optimizer to use'
    )
    parser.add_argument(
        '-O', '--optargs', type=float, nargs='+', default=[],
        help='named arguments put into the optimizer constructor'
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
        '-D', '--dtl', type=int, choices=[0, 1, 2], default=2,
        help='what kind of transfer learning, 0: no dtl, ' +
             '1: take disc pretrained on same dataset, ' +
             '2: take disc pretrained on other dataset'
    )
    parser.add_argument(
        '-c', '--check_args', action='store_true',
        help='add flag to print the argparse result'
    )

    args = parser.parse_args()
    if args.check_args:
        input(args)

    main(args)

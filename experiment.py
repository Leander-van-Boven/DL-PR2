import csv
from os import path

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from keras.models import Model


def combine_model(gen, disc, latent_dim):
    """Combines a generator and discriminator model into a GAN.

    Parameters
    ----------
    gen : Model
        The generator model.
    disc : Model
        The discriminator model.
    latent_dim : tuple
        The dimension of the noise vector input for the generator.

    Returns
    -------
    Model
        The generator and discriminator combined into a Model as GAN.
    """
    noise = Input(shape=latent_dim)
    img = gen(noise)
    disc.trainable = False
    valid = disc(img)
    return Model(noise, valid)


def run_experiment(gen, disc, x_train, opt, epochs, batch_size,
                   latent_dim, log_path, img_path, log_interval, print):
    """Main experiment function. Runs the epochs and makes sure that
        everything is logged.

    Parameters
    ----------
    gen : Model
        The generator model.
    disc : Model
        The discriminator model.
    x_train : numpy-array
        The training data.
    opt : str/keras.Optimizer
        The optimizer to use when compiling the combined GAN model.
    epochs : int
        The amount of epochs to run the model with.
    batch_size : int
        The batch size used when training the models.
    latent_dim : int
        The size of the noise vector of the generator.
    log_path : str
        The directory in which to log the results.
    img_path : str
        The directory in which to save the generated images.
    log_interval : float
        Determines factor of epochs in which the results are logged.
    print : function
        The function to use to print the results to the command line.
        If 'print' is 'Nothing' then nothing will be printed.
    """
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Create the combined GAN
    disc.trainable = False
    combined = combine_model(gen, disc, latent_dim)
    combined.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # combined.summary()

    log_file = path.join(log_path, "training.csv")

    def save_imgs(epoch):
        """Generates 25 images with the GAN and saves them.

        Parameters
        ----------
        epoch : int
            The current epoch of the experiment, used for the file name.
        """
        # Generate 25 images
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = gen.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Plot images 5 by 5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        # Save images
        fig.savefig(f"{img_path}/{epoch:05}.png")
        plt.close()

    # Write results to csv file
    with open(log_file, mode='a+') as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(
            ["epoch", "d_loss_real", "d_loss_fake", "d_loss", "d_acc_real",
             "d_acc_fake", "d_acc", "g_loss", "g_acc"]
        )

    for epoch in range(epochs):
        # Select training images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        # Generate images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = gen.predict(noise)

        # Train the discriminator
        d_loss_real = disc.train_on_batch(imgs, valid)
        d_loss_fake = disc.train_on_batch(gen_imgs, fake)

        # Train generator
        g_loss = combined.train_on_batch(noise, valid)

        # If applicable, print results
        print("\n--EPOCH %s--" % epoch)
        print("d_real, %s" %
              list(map(str, zip(disc.metrics_names, d_loss_real))))
        print("d_fake, %s" %
              list(map(str, zip(disc.metrics_names, d_loss_fake))))
        print("g     , %s" %
              list(map(str, zip(combined.metrics_names, g_loss))))

        # Write results to csv file
        with open(log_file, mode='a+') as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=',', quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )

            csv_writer.writerow(
                [epoch, d_loss_real[0], d_loss_fake[0],
                 (d_loss_real[0] + d_loss_fake[0]) /
                 2, d_loss_real[1], d_loss_fake[1],
                 (d_loss_real[1] + d_loss_fake[1]) / 2, g_loss[0], g_loss[1]]
            )

        # Save some generator images if we are in a correct epoch
        if log_interval > 0 and epoch % log_interval == 0:
            save_imgs(epoch)

    # Save final images
    if log_interval > 0:
        save_imgs(epochs)

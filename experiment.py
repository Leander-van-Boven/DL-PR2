import numpy as np
import os
import datetime
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from gan import combine_model


def run_experiment(gen, disc, x_train, opt, epochs, batch_size, latent_dim, log_dir):
    # Rescale X_train to [-1, 1]?

    noise_size = latent_dim

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    disc.trainable = False
    combined = combine_model(gen, disc, latent_dim)
    combined.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    combined.summary()

    # prepare log output file. files start with YYYY-MM-DDTHH:MM
    run_time = datetime.datetime.now().isoformat(timespec='minutes')
    run_time = run_time.replace(':', '-')
    # might wanna include some details about the run here as well so we can
    # identify runs easily. does python have a nameof() operator?
    log_file_name = run_time + '-training.log'

    log_file = os.path.join(log_dir, log_file_name)

    def save_imgs(epoch):
        os.makedirs('../images', exist_ok=True)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = gen.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])  # , cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../images/img_%d.png" % epoch)
        plt.close()

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_size))
        gen_imgs = gen.predict(noise)

        d_loss_real = disc.train_on_batch(imgs, valid)
        d_loss_fake = disc.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        g_loss = combined.train_on_batch(noise, valid)
        print(f"epoch: {epoch}\t d_loss_real: {d_loss_real[0]:.5f}\t d_loss_fake: {d_loss_fake[0]:.5f}\t g_loss: {g_loss[0]:.5f}")

        # write results to csv file
        with open(log_file, mode='a+') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow([epoch, d_loss[0], 100*d_loss[1], g_loss[0]])

        if epoch % 10 == 0:
            save_imgs(epoch)

        # print(type(epoch), type(d_loss[0]), type(100*d_loss[1]), type(g_loss[0]))
        # print(g_loss[0], g_loss[1])

        # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
        #       (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

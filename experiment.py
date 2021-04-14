import numpy as np
import os
import datetime
import csv
from numpy.core import shape_base
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from gan import combine_model


def run_experiment(gen, disc, x_train, opt, epochs, batch_size,
                   latent_dim, log_path, img_path, log_interval, print):

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
    # combined.summary()

    log_file = os.path.join(log_path, "training.csv")

    def save_imgs(epoch):
        # os.makedirs('../images', exist_ok=True)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = gen.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1


        # filename = os.path.join(, "", , ".png")
        fig.savefig(f"{img_path}/{epoch:05}.png")
        plt.close()

     # write results to csv file
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
        # Train discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_size))
        gen_imgs = gen.predict(noise)

        # fig, axs = plt.subplots(1, 3)
        # im = np.random.randint(imgs.shape[0])
        # for i in range(3):
        #     axs[i].imshow(gen_imgs[im, :, :, i], cmap='gray')
        # plt.show()

        d_loss_real = disc.train_on_batch(imgs, valid)
        d_loss_fake = disc.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_acc = 100*d_loss[1]

        # Train generator
        g_loss = combined.train_on_batch(noise, valid)
        # print(f"epoch: {epoch}\td_loss_real: {d_loss_real[0]:.5f}\t" +
        #       f"d_loss_fake: {d_loss_fake[0]:.5f}\t")
        # print(f"\t\td_acc: {d_acc:.5f}\t g_loss: {g_loss[0]:.5f}")
        # print("")

        print("\n--EPOCH %s--" % epoch)
        print("d_real, %s" % list(map(str, zip(disc.metrics_names, d_loss_real))))
        print("d_fake, %s" % list(map(str, zip(disc.metrics_names, d_loss_fake))))
        print("g     , %s" % list(map(str, zip(combined.metrics_names, g_loss))))

        # write results to csv file
        with open(log_file, mode='a+') as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=',', quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )

            csv_writer.writerow(
                [epoch, d_loss_real[0], d_loss_fake[0], 
                (d_loss_real[0]+d_loss_fake[0])/2, d_loss_real[1], d_loss_fake[1],
                (d_loss_real[1]+d_loss_fake[1])/2, g_loss[0], g_loss[1]]
            )

        if log_interval > 0 and epoch % log_interval == 0:
            save_imgs(epoch)

    if log_interval > 0:
        save_imgs(epochs)

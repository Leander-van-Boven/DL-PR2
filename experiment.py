import numpy as np
import os
import datetime

from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from gan import combine_model


def run_experiment(gen, disc, X_train, opt, epochs, batch_size, log_dir):
    # Rescale X_train to [-1, 1]?

    noise_size = gen.layers[0].shape

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    combined = combine_model(gen, disc)
    combined.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # prepare log output file. files start with YYYY-MM-DDTHH:MM
    run_time = datetime.datetime.now().isoformat(timespec='minutes')
    # might wanna include some details about the run here as well so we can
    # identify runs easily. does python have a nameof() operator?
    log_file_name = run_time + '-training.log'

    log_file = os.path.join(log_dir, log_file_name)

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_size))
        gen_imgs = gen.predict(noise)

        d_loss_real = disc.train_on_batch(imgs, valid)
        d_loss_fake = disc.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        g_loss = combined.train_on_batch(noise, valid)

        # TODO: write logs to log_file somewhere around here. do we want to use
        #       a plain csv writer or a keras callback?

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
              (epoch, d_loss[0], 100*d_loss[1], g_loss))

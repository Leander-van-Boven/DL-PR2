#####################################################################
# THIS FILE HAS BEEN COPIED FROM THE eriklindernoren/Keras-GAN REPO #
# then some modifications were made                                 #
#####################################################################

from __future__ import print_function, division
from gan import build_discriminator

from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0

from set_session import initialize_session


class DCGAN():
    def __init__(self):
        # Input shape
        self.dataset = "mnist"
        self.discfunc = self.build_discriminator2

        if self.dataset == "mnist":
            if self.discfunc == self.build_discriminator2:
                self.img_rows = self.img_cols = 32
            else:
                self.img_rows = self.img_cols = 28
            self.channels = 1
        elif self.dataset == "cifar10":
            self.img_rows = 32
            self.img_cols = 32
            self.channels = 3
        else:
            exit()
        # self.img_rows = 32
        # self.img_cols = 32
        # self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.discfunc()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        print('img.shape', img.shape)
        if (self.dataset == "mnist" and
            self.discfunc == self.build_discriminator2):
            img = tf.image.grayscale_to_rgb(img)
            print('img.shape after', img.shape)


        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        if (self.dataset == "mnist" and
            self.discfunc != self.build_discriminator2):
            model.add(Dense(128 * 7 * 7, activation="relu",
                            input_dim=self.latent_dim))
            model.add(Reshape((7, 7, 128)))
        else:
            model.add(Dense(128 * 8 * 8, activation="relu",
                            input_dim=self.latent_dim))
            model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_generator2(self):
        model = Sequential()

        if self.dataset == "mnist":
            model.add(Dense(256 * 7 * 7, activation="relu",
                            input_dim=self.latent_dim))
            model.add(Reshape((7, 7, 256)))
        else:
            model.add(Dense(256 * 8 * 8, activation="relu",
                            input_dim=self.latent_dim))
            model.add(Reshape((8, 8, 256)))
        # model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_generator3(self):
        model = Sequential()

        if self.dataset == "mnist":
            model.add(Dense(512 * 7 * 7, activation="relu",
                            input_dim=self.latent_dim))
            model.add(Reshape((7, 7, 512)))
        else:
            model.add(Dense(512 * 8 * 8, activation="relu",
                            input_dim=self.latent_dim))
            model.add(Reshape((8, 8, 512)))
        # model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2,
                         input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_discriminator3(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2,
                         input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(1028, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_discriminator2(self):
        cnn_disc = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling=None
            # classifier_activation="softmax"
        )
        # cnn_disc.trainable = False
        flattened = Flatten()(cnn_disc.output)
        bool_layer = Dense(1, activation='sigmoid')(flattened)
        disc = Model(inputs=cnn_disc.input, outputs=bool_layer)
        disc.summary()
        return disc

    def train(self, epochs=4000, batch_size=32, save_interval=25):
        # Load the dataset
        if self.dataset == "mnist":
            (X_train, _), (_, _) = mnist.load_data()
            X_train = np.expand_dims(X_train, axis=-1)
        else:
            (X_train, _), (_, _) = cifar10.load_data()


        # X_train = np.expand_dims(X_train, axis=3)

        print('X_train', X_train.shape)
        # print('X_train head', X_train[:10])

        if (self.dataset == "mnist" and
            self.discfunc == self.build_discriminator2):
            X_train = tf.convert_to_tensor(X_train, dtype=tf.uint8)
            X_train = tf.image.pad_to_bounding_box(X_train, 2, 2, 32, 32)
            X_train = tf.image.grayscale_to_rgb(X_train)
            X_train = np.array(X_train)
            print('X_train after', X_train.shape)

        #noise = tf.random.normal(
        #    shape=tf.shape(X_train), mean=0.0, stddev=1, dtype=tf.float32
        #)
        #print(np.max(noise), np.min(noise))
        #print(np.max(X_train), np.min(X_train))
        #X_train = tf.add(X_train, noise)

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            if (self.dataset == "mnist" and
                self.discfunc == self.build_discriminator2):
                 gen_imgs = tf.convert_to_tensor(gen_imgs)
                 gen_imgs = tf.image.grayscale_to_rgb(gen_imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # yield epoch
                self.save_imgs(epoch)

    def initialize_plot(self):
        image = np.random.normal(0, 1, self.img_shape)

        self.fig, self.axs = plt.subplots()
        # self.imgs = [None] * (r * c)
        #cnt = 0
        #for i in range(r):
        #    for j in range(c):
        #        if self.dataset == "mnist":
        #            self.imgs[cnt] = self.axs[i, j].imshow(image[:, :, 0], cmap='gray')
        #        else:
        #            self.imgs[cnt] = self.axs[i, j].imshow(images[:, :, :])
        #        self.axs[i, j].axis('off')
        #        cnt += 1
        if self.dataset == "mnist":
            self.img = self.axs.imshow(image.squeeze(), cmap='gray')
        else:
            self.img = self.axs.imshow(image)
        self.axs.axis('off')

    def init_plot(self):
        image = np.random.normal(0, 1, self.img_shape)
        if self.dataset == "mnist":
            self.img.set_data(image.squeeze())
            self.img.set_cmap('gray')
        else:
            self.img.set_data(image)
        return [self.img]

    def update_plot(self, epoch):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        if self.dataset == "mnist":
            self.img.set_data(gen_imgs[0, :, :, 0])
            self.img.set_cmap('gray')
        else:
            self.img.set_data(gen_imgs[0, :, :, :])
        return [self.img]

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.dataset == "mnist":
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                else:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("../images/%s_%d.png" % (self.dataset,epoch))
        plt.close()

    def start_animation(self):
        self.initialize_plot()
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, self.train,
            blit=True, interval=1, repeat=False,
            init_func=self.init_plot)
        plt.show()


if __name__ == '__main__':
    initialize_session()
    dcgan = DCGAN()
    # dcgan.start_animation()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
from keras.callbacks import EarlyStopping
from keras.datasets import mnist, fashion_mnist
from keras.layers import Dense
from keras.metrics import Precision
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical

from dcgan1 import build_discriminator1
from dcgan2 import build_discriminator2
from set_session import initialize_session

from sys import exit


def train_disc(path, dataset, disc_arch):
    """Trains a discriminator on the given dataset
        and saves it to the given folder.

    Parameters
    ----------
    path : str
        The folder in which to save the (trained) model.
    dataset : numpy-array
        The data on which to train the discriminator.
    disc_arch : function
        Function to use to build the discriminator model.
    """
    # Import dataset and apply pre-processing
    (X_train, y_train), (X_test, y_test) = dataset.load_data()
    X_train = (X_train / 127.5) - 1.
    y_train = to_categorical(y_train)
    X_test = (X_test / 127.5) - 1.
    y_test = to_categorical(y_test)

    # Build discriminator model and compile
    disc = disc_arch(img_shape=(28, 28, 1),
                     include_dense=False, compile=False)
    out = Dense(10, 'softmax')(disc.output)
    model = Model(disc.input, out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[Precision(), 'accuracy'])

    # Create early stopping callback and train discriminator
    es = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=1000,
        verbose=1,
        shuffle=True,
        validation_split=.1,
        callbacks=[es])

    # Prepare discriminator for saving and save it
    disc.trainable = False
    bool_out = Dense(1, 'sigmoid')(disc.output)
    final_model = Model(disc.input, bool_out)
    final_model.save(path)


if __name__ == "__main__":
    initialize_session()

    # Define the discriminator models with arguments
    models = [
        ["./discriminator1_digits", mnist, build_discriminator1],
        ["./discriminator2_digits", mnist, build_discriminator2],
        ["./discriminator1_fashion", fashion_mnist, build_discriminator1],
        ["./discriminator2_fashion", fashion_mnist, build_discriminator2]
    ]
    # Train each discriminator model
    for model in models:
        print(model[0])
        train_disc(*model)

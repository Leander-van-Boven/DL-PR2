import tensorflow as tf


def initialize_session():
    devs = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devs[0], True)

    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    _ = tf.compat.v1.Session(config=config)

# devs = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devs[0], True)
# print('MEMORY GROWTH:', tf.config.experimental.get_memory_growth(devs[0]))

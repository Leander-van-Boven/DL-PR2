from keras_gan import dcgan

# Combines generator architecture with discriminator architecture
# Contains logic to train and evaluate the GAN

# class GAN:
#     def __init__(self, gen, desc):
#         pass

#     def train(self):
#         pass

#     def evaluate(self):
#         pass

def build_gan(desc_args):
    dc = DCGAN()
    dc.discriminator = build_discriminator(desc_args)
    return dc


def build_discriminator(**args):
    return None

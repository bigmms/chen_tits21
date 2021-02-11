from __future__ import print_function, division
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import datetime
from utility_file.data_loader import DataLoader
from utility_file.model import discriminator_model, generator_model
from utility_file.common import *
from datetime import datetime
import os
from skimage.io import imsave

class Glare_Removal():
    def __init__(self):
        self.img_rows = 80
        self.img_cols = 176
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'dataset'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, int(patch * 2.2), 1)

        # Number of filters in the first layer of G and D
        optimizer = Adam(0.0002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer)

        # Input images
        I = Input(shape=self.img_shape)
        self.generator = self.build_generator()
        self.generator.load_weights('./save_weight/generator.h5')
        L_prime, B_prime, B, Clean= self.generator([I])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([I, L_prime, B_prime,Clean])

        self.combined = Model(inputs=[I], outputs=[valid, L_prime, B_prime, B,Clean])
        self.combined.compile(
            loss=['mse', 'mse', 'mse', 'mse', 'mse'],
            loss_weights=[1, 100, 100, 100, 100], optimizer=optimizer)

    def build_generator(self):
        return generator_model(self.img_shape)

    def build_discriminator(self):
        return discriminator_model(input_shape=self.img_shape, filters=64)

    def test(self):
        os.makedirs('./test_results/', exist_ok=True)
        I = self.data_loader.load_test_data()

        start_time = datetime.now()
        for b in range(len(I)):
            I_batch = I[b:(b + 1)]
            L_prime, B_prime, B, Clean = self.generator.predict([I_batch])
            output = Clean[0].astype(np.uint8)
            imsave("./test_results/%05d.png" % (b + 1), output)
            print(b + 1)
        end_time = (datetime.now() - start_time) / len(I)

        print("Avg Time: ", end_time)


if __name__ == '__main__':
    # training model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    glare_removal_model = Glare_Removal()
    glare_removal_model.test()

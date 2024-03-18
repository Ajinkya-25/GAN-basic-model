import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(img, lbl), (_, _) = tf.keras.datasets.mnist.load_data()
img = img.reshape(img.shape[0], 28, 28, 1)
img = img[:100] / 255.0

buffer_size = 600

traindata = tf.data.Dataset.from_tensor_slices(img)
traindata = traindata.shuffle(buffer_size).batch(256)

done = 0


def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(256,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))  # Removed 'autocast' argument

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = generator_model()
discriminator = discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 128


def train_loop(EPOCHS, img, done):
    for epoch in range(EPOCHS):
        print(epoch)
        for real_image in img:
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                real_images = tf.reshape(real_image, [-1, 28, 28, 1]) / 255.0
                fake_image = generator(noise, training=True)
                real_output = discriminator(real_images, training=True)  # Corrected input
                fake_output = discriminator(fake_image, training=True)
                disc_loss = discriminator_loss(real_output, fake_output)
                gen_loss = generator_loss(fake_output)

            gen_gradienttape = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradienttape = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            done += 1
            generator_optimizer.apply_gradients(zip(gen_gradienttape, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradienttape, discriminator.trainable_variables))
            if done % 100 == 0:
                print("img done")


def generate_and_save_images(model, epoch, test_input):
    print("in images ")
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


test_input = np.random.normal(0, 1, (num_examples_to_generate, 256)).astype(np.float32)
num_examples_to_generate = 10
noise_dim = 256

train_loop(EPOCHS, img, done)
print("train done")
generate_and_save_images(generator, EPOCHS, test_input)

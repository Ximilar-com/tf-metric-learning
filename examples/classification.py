import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import copy

from tf_metric_learning.layers import ProxyAnchorLoss
from tf_metric_learning.utils.projector import TBProjectorCallback
from tf_metric_learning.utils.recall import AnnoyEvaluatorCallback
from tf_metric_learning.utils.constants import *

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60


@tf.function
def load_img_train(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32)  # converts to 0-1
    image = tf.image.resize(image, [IMG_SIZE + 30, IMG_SIZE + 30])
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def filter_train(image, label):
    return label < 98


@tf.function
def load_img_test(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label


def filter_test(image, label):
    return label >= 98


def normalize_images(images):
    return images / 255.0


def scheduler(epoch):
    if epoch < 20:
        return 0.001
    elif epoch < 40:
        return 0.0001
    return 0.00001


(ds_train, ds_test), ds_info = tfds.load(
    "cars196", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=True
)

# filter the train and test set by the classes
ds_to_train = ds_train.concatenate(ds_test).filter(filter_train)
ds_to_test = ds_test.concatenate(ds_train).filter(filter_test)

ds_to_train = ds_to_train.map(load_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_to_train = ds_to_train.shuffle(3000)
ds_to_train = ds_to_train.batch(BATCH_SIZE).prefetch(10)

ds_to_test = ds_to_test.map(load_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_images, test_labels = [], []
for image, label in tfds.as_numpy(ds_to_test):
    # for projector we need to convert the image from 0-1 back to 0-255
    image = np.squeeze(copy.deepcopy(image)) * 255.0
    test_images.append(image)
    test_labels.append(label)

test_images = np.asarray(test_images)
test_labels = np.squeeze(test_labels)

ds_to_test = ds_to_test.shuffle(len(test_images)).cache().batch(BATCH_SIZE)

embedding_size, num_class, num_centers = 256, int(196 / 2), 10
input_shape = (IMG_SIZE, IMG_SIZE, 3)

# define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units=embedding_size, activation="relu")(dropout)
base_network = tf.keras.Model(inputs=inputs, outputs=embeddings)

# classification model
dropout_2 = tf.keras.layers.Dropout(0.25)(embeddings)
output_tensor = tf.keras.layers.Dense(num_class, activation="softmax", name="probs", kernel_initializer="he_uniform")(
    dropout_2
)

# define the model and compile it
model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
model.summary()

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=scheduler(0)), metrics=["sparse_categorical_accuracy"])

# callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tb")

projector = TBProjectorCallback(
    base_network,
    "tb/projector",
    test_images,
    test_labels,
    batch_size=BATCH_SIZE,
    image_size=64,
    normalize_fn=normalize_images,
    normalize_eb=True,
)

# callback for recall evaluation
divide = int(len(test_images) / 2)
evaluator = AnnoyEvaluatorCallback(
    base_network,
    {"images": test_images[:divide], "labels": test_labels[:divide]},
    {"images": test_images[divide:], "labels": test_labels[divide:]},
    normalize_fn=normalize_images,
    normalize_eb=True,
    eb_size=embedding_size,
    freq=5,
)

model.fit(ds_to_train, callbacks=[tensorboard, evaluator, scheduler_cb, projector], epochs=EPOCHS)

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
    return {"images": image, "labels": label}, label


@tf.function
def load_img_test(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return {"images": image, "labels": label}, label


def scheduler(epoch):
    if epoch < 20:
        return 0.0001
    elif epoch < 40:
        return 0.00001
    return 0.000001


(ds_train, ds_test), ds_info = tfds.load(
    "cars196", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=True
)

# filter the train and test set by the classes
ds_to_train = ds_train.concatenate(ds_test).filter(lambda image, label: label < 98)
ds_to_test = ds_test.concatenate(ds_train).filter(lambda image, label: label >= 98)

ds_to_train = ds_to_train.map(load_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_to_train = ds_to_train.shuffle(3000)
ds_to_train = ds_to_train.batch(BATCH_SIZE)

ds_to_test = ds_to_test.map(load_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_images, test_labels = [], []
for data, label in tfds.as_numpy(ds_to_test):
    # for projector we need to convert the image from 0-1 back to 0-255
    image = np.squeeze(copy.deepcopy(data["images"])) * 255.0
    test_images.append(image)
    test_labels.append(label)

test_images = np.asarray(test_images)
test_labels = np.squeeze(test_labels)

ds_to_test = ds_to_test.shuffle(len(test_images)).cache().batch(BATCH_SIZE)

embedding_size, num_class, num_centers = 64, int(196 / 2), 10
input_shape = (IMG_SIZE, IMG_SIZE, 3)

# define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units=embedding_size)(dropout)
base_network = tf.keras.Model(inputs=inputs, outputs=embeddings)

# define the input and output tensors
input_label = tf.keras.layers.Input(shape=(1,), name="labels")
output_tensor = ProxyAnchorLoss(num_class, embedding_size)({EMBEDDINGS: base_network.outputs[0], LABELS: input_label})

# define the model and compile it
model = tf.keras.Model(inputs=[inputs, input_label], outputs=output_tensor)
model.summary()

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=scheduler(0)))

# callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tb")

projector = TBProjectorCallback(
    base_network,
    "tb/projector",
    test_images,
    test_labels,
    batch_size=BATCH_SIZE,
    image_size=64,
    normalize_fn=lambda images: images / 255.0,
    normalize_eb=True,
)

# callback for recall evaluation
divide = int(len(test_images) / 2)
evaluator = AnnoyEvaluatorCallback(
    base_network,
    {"images": test_images[:divide], "labels": test_labels[:divide]},
    {"images": test_images[divide:], "labels": test_labels[divide:]},
    normalize_fn=lambda images: images / 255.0,
    normalize_eb=True,
    eb_size=embedding_size,
    freq=5,
)

model.fit(ds_to_train, callbacks=[tensorboard, evaluator, scheduler_cb, projector], epochs=EPOCHS)

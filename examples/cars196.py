import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import copy

from tf_metric_learning.layers import ProxyAnchorLoss, SoftTripleLoss
from tf_metric_learning.utils.projector import TBProjectorCallback

IMG_SIZE = 128
BATCH_SIZE = 100

def load_img_train(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32) # converts to 0-1
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # image = tf.image.resize(image, [IMG_SIZE + 30, IMG_SIZE + 30])
    # image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_hue(image, 0.1)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    return {"images": image, "labels": label}, label

def load_img_test(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32) # converts to 0-1
    image = tf.image.resize(image, [IMG_SIZE , IMG_SIZE])
    return {"images": image, "labels": label}, label

def normalize_images(images):
    return images/255.0

(ds_train, ds_test), ds_info = tfds.load(
    'cars196',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.map(
    load_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(load_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_images, test_labels = [], []
for data , label in tfds.as_numpy(ds_test):
    # for projector we need to convert the image back to 0-255
    image = np.squeeze(data["images"])*255.0
    test_images.append(image)
    test_labels.append(label)

ds_test = ds_test.shuffle(ds_info.splits['test'].num_examples).batch(BATCH_SIZE)

embedding_size, num_class, num_centers = 256, 196, 4
input_shape = (IMG_SIZE, IMG_SIZE, 3)

# define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet", layers=tf.keras.layers)(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units = embedding_size)(dropout)
base_network = tf.keras.Model(inputs = inputs, outputs = embeddings)

# define the input and output tensors
input_label = tf.keras.layers.Input(shape=(1,), name="labels")
# output_tensor = ProxyAnchorLoss(num_class, embedding_size)(base_network.outputs[0], input_label)
output_tensor = SoftTripleLoss(num_class, num_centers, embedding_size)(base_network.outputs[0], input_label)

# define the model and compile it
model = tf.keras.Model(inputs=[inputs, input_label], outputs=output_tensor)
model.compile(optimizer="adam")

# create simple callback for projecting embeddings after every epoch
projector = TBProjectorCallback(
    base_network,
    "tb",
    np.asarray(test_images),
    np.squeeze(test_labels),
    batch_size=BATCH_SIZE,
    image_size=64,
    normalize_fn=normalize_images,
    normalize_eb=True,
)

model.fit(
    ds_train,
    validation_data=ds_test,
    callbacks=[projector],
    epochs=20
)

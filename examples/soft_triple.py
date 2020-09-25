import tensorflow as tf
import numpy as np
import copy

from tf_metric_learning.layers import SoftTripleLoss
from tf_metric_learning.utils.projector import TBProjectorCallback
from tf_metric_learning.utils.recall import AnnoyEvaluatorCallback
from tf_metric_learning.utils.constants import *


def normalize_images(images):
    return images / 255.0


# load data images
BATCH_SIZE = 32
embedding_size, num_class, num_centers = 64, 10, 10
input_shape = (32, 32, 3)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units=embedding_size)(dropout)
base_network = tf.keras.Model(inputs=inputs, outputs=embeddings)

# define the input and output tensors
input_label = tf.keras.layers.Input(shape=(1,), name="labels")
output_tensor = SoftTripleLoss(num_class, num_centers, embedding_size)(
    {EMBEDDINGS: base_network.outputs[0], LABELS: input_label}
)

# define the model and compile it
model = tf.keras.Model(inputs=[inputs, input_label], outputs=output_tensor)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001))

train_data = {"images": normalize_images(train_images), "labels": train_labels}

validation_data = {"images": normalize_images(test_images), "labels": test_labels}

# callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tb")

projector = TBProjectorCallback(
    base_network,
    "tb/projector",
    copy.deepcopy(test_images),
    np.squeeze(test_labels),
    batch_size=BATCH_SIZE,
    image_size=32,
    normalize_fn=normalize_images,
    normalize_eb=True,
    freq=1,
)

evaluator = AnnoyEvaluatorCallback(
    base_network,
    {"images": validation_data["images"][:5000], "labels": np.squeeze(validation_data["labels"][:5000])},
    {"images": validation_data["images"][5000:], "labels": np.squeeze(validation_data["labels"][5000:])},
    normalize_eb=True,
    eb_size=embedding_size,
    progress=False,
)

model.fit(
    train_data,
    train_labels,
    validation_data=(validation_data, test_labels),
    callbacks=[tensorboard, evaluator, projector],
    shuffle=True,
    epochs=20,
    batch_size=BATCH_SIZE,
)

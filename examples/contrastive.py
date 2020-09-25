import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import copy

from tf_metric_learning.layers.contrastive import ContrastiveLoss
from tf_metric_learning.miners.annoy import TripletAnnoyMiner
from tf_metric_learning.utils.projector import TBProjectorCallback
from tf_metric_learning.utils.recall import AnnoyEvaluatorCallback
from tf_metric_learning.utils.constants import *

from base import store_images, BaseMinerSequence

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60


class ContrastiveSequence(BaseMinerSequence):
    def __init__(self, base_model, train_images, train_labels, embedding_size, batch_size):
        super().__init__(base_model, train_images, train_labels, embedding_size, batch_size)

    def __getitem__(self, idx):
        anchors, positives, labels = [], [], []
        base_index = idx * BATCH_SIZE

        for i in range(BATCH_SIZE):
            id_image = self.indexes[base_index + i]
            anchor_image, anchor_label = self.train_images[id_image], self.train_labels[id_image]
            anchor_image = self.augment_image(anchor_image).numpy()

            # pick the positive image for anchor in triplet
            positive_id = self.pick_positive(anchor_image, anchor_label, id_image)
            if positive_id is None:
                continue
            positive, positive_label = self.train_images[positive_id], self.train_labels[positive_id]
            positive = self.augment_image(positive).numpy()

            # pick the negative image for anchor in triplet
            negative_id = self.pick_negative(anchor_image, anchor_label, id_image)
            if negative_id is None:
                continue
            negative, negative_label = self.train_images[negative_id], self.train_labels[negative_id]
            negative = self.augment_image(negative).numpy()

            store_images("test.jpg", anchor_image, positive, negative)

            # add them to the batch
            anchors.append(anchor_image)
            positives.append(positive)
            labels.append(1.0)
            anchors.append(anchor_image)
            positives.append(negative)
            labels.append(0.0)

        return [np.asarray(anchors), np.asarray(positives), np.asarray(labels)]


@tf.function
def load_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32)  # converts to 0-1
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label


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

ds_to_train = ds_to_train.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_images, train_labels = [], []
for image, label in tfds.as_numpy(ds_to_train):
    train_images.append(image)
    train_labels.append(label)

train_images = np.asarray(train_images)
train_labels = np.squeeze(train_labels)

ds_to_test = ds_to_test.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_images, test_labels = [], []
for image, label in tfds.as_numpy(ds_to_test):
    # for projector we need to convert the image from 0-1 back to 0-255
    image = np.squeeze(copy.deepcopy(image)) * 255.0
    test_images.append(image)
    test_labels.append(label)

test_images = np.asarray(test_images)
test_labels = np.squeeze(test_labels)

ds_to_test = ds_to_test.shuffle(len(test_images)).cache().batch(BATCH_SIZE)

embedding_size, num_class, num_centers = 128, int(196 / 2), 10
input_shape = (IMG_SIZE, IMG_SIZE, 3)

# define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units=embedding_size)(dropout)
base_network = tf.keras.Model(inputs=inputs, outputs=embeddings)

input_anchor = tf.keras.Input(shape=input_shape, name="input_anchor")
input_second = tf.keras.Input(shape=input_shape, name="input_second")
input_labels = tf.keras.Input(shape=(1,), name="input_labels")

# this will create three networks with shared weights ...
net_anchor = base_network(input_anchor)
net_second = base_network(input_second)

loss_layer = ContrastiveLoss(margin=1.0, normalize=True)(
    {ANCHOR: net_anchor, POSITIVE: net_second, LABELS: input_labels}
)
model = tf.keras.Model(inputs=[input_anchor, input_second, input_labels], outputs=loss_layer)

# create simple callback for projecting embeddings after every epoch
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
    freq=1,
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
    freq=1,
)

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=scheduler(0)))

model.fit(
    ContrastiveSequence(base_network, train_images, train_labels, embedding_size, BATCH_SIZE),
    callbacks=[tensorboard, scheduler_cb, evaluator, projector],
    epochs=EPOCHS,
)

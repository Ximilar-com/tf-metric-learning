import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import copy
import random
import cv2

from tf_metric_learning.layers.triplet import TripletLoss
from tf_metric_learning.miners.annoy import TripletAnnoyMiner
from tf_metric_learning.utils.projector import TBProjectorCallback
from tf_metric_learning.utils.recall import AnnoyEvaluatorCallback


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60

def store_images(filename, *images):
    new_images = [cv2.cvtColor(image * 255.0, cv2.COLOR_RGB2BGR) for image in images]

    if random.random() < 0.1:
        cv2.imwrite(filename, np.concatenate(new_images, axis=1))


class AnchorPositiveNegative(tf.keras.utils.Sequence):
    def __init__(self, base_model, train_images, train_labels):
        self.base_model = base_model
        self.train_images = train_images
        self.train_labels = train_labels

        self.mapping = self.create_labels_id_data(self.train_images, self.train_labels)
        self.miner = TripletAnnoyMiner(base_model, embedding_size, train_labels, progress=True)
        self.indexes = np.arange(len(train_images))
        self.epoch = 0
        self.on_epoch_end()

    def create_labels_id_data(self, train_images, train_labels):
        data = {}
        for i, (image, label) in enumerate(zip(train_images, train_labels)):
            if label not in data:
                data[label] = [i]
            else:
                data[label].append(i)
        return data

    def __len__(self):
        return len(self.train_images) // BATCH_SIZE

    def on_epoch_end(self):
        embeddings = self.base_model.predict(self.train_images)
        self.miner.reindex(embeddings)
        np.random.shuffle(self.indexes)
        self.epoch += 1

    def pick_positive(self, anchor_image, anchor_label, id):
        return random.choice(self.mapping[anchor_label])

    def pick_negative(self, anchor_image, anchor_label, id):
        if self.epoch < 10:
            # randomly pick easy negative image
            numbers = list(range(0, len(self.mapping.keys())))
            numbers.remove(anchor_label)
            random_label = random.choice(numbers)
            return random.choice(self.mapping[random_label])
        elif self.epoch < 20:
            # in second ten epochs, pick the easiest one for learning
            return self.miner.search_easiest_negative(self.miner.get_item_vector(id), anchor_label, n=50)
        return self.miner.search_hardest_negative(self.miner.get_item_vector(id), anchor_label, n=50)

    def __getitem__(self, idx):
        anchors, positives, negatives = [], [], []
        base_index = idx * BATCH_SIZE

        for i in range(BATCH_SIZE):
            id_image = self.indexes[base_index +i]
            anchor_image, anchor_label = self.train_images[id_image], self.train_labels[id_image]

            # pick the positive image for anchor in triplet
            positive_id = self.pick_positive(anchor_image, anchor_label, id_image)
            if positive_id is None:
                continue
            positive, positive_label = self.train_images[positive_id], self.train_labels[positive_id]

            # pick the hardest negative image for anchor in triplet
            negative_id = self.pick_negative(anchor_image, anchor_label, id_image)
            if negative_id is None:
                continue
            negative, negative_label = self.train_images[negative_id], self.train_labels[negative_id]

            store_images("test.jpg", anchor_image, positive, negative)

            # add them to the batch
            anchors.append(anchor_image)
            positives.append(positive)
            negatives.append(negative)

        return [np.asarray(anchors), np.asarray(positives), np.asarray(negatives)]


@tf.function
def load_img_train(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32) # converts to 0-1
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # image = tf.image.resize(image, [IMG_SIZE + 30, IMG_SIZE + 30])
    # image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_hue(image, 0.1)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def filter_train(image, label):
    return label < 98

@tf.function
def load_img_test(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE , IMG_SIZE])
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
    'cars196',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# filter the train and test set by the classes
ds_to_train = ds_train.concatenate(ds_test).filter(filter_train)
ds_to_test = ds_test.concatenate(ds_train).filter(filter_test)

ds_to_train = ds_to_train.map(
    load_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_images, train_labels = [], []
for image , label in tfds.as_numpy(ds_to_train):
    train_images.append(image)
    train_labels.append(label)

train_images = np.asarray(train_images)
train_labels = np.squeeze(train_labels)

ds_to_test = ds_to_test.map(load_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_images, test_labels = [], []
for image , label in tfds.as_numpy(ds_to_test):
    # for projector we need to convert the image from 0-1 back to 0-255
    image = np.squeeze(copy.deepcopy(image))*255.0
    test_images.append(image)
    test_labels.append(label)

test_images = np.asarray(test_images)
test_labels = np.squeeze(test_labels)

ds_to_test = ds_to_test.shuffle(len(test_images)).cache().batch(BATCH_SIZE)

embedding_size, num_class, num_centers = 128, int(196/2), 10
input_shape = (IMG_SIZE, IMG_SIZE, 3)

# define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units = embedding_size)(dropout)
base_network = tf.keras.Model(inputs = inputs, outputs = embeddings)

input_anchor = tf.keras.Input(shape=input_shape, name='input_anchor')
input_positive = tf.keras.Input(shape=input_shape, name='input_pos')
input_negative = tf.keras.Input(shape=input_shape, name='input_neg')

# this will create three networks with shared weights ...
net_anchor = base_network(input_anchor)
net_positive = base_network(input_positive)
net_negative = base_network(input_negative)

loss_layer = TripletLoss()(net_anchor, net_positive, net_negative)
triplet_model = tf.keras.Model(inputs = [input_anchor, input_positive, input_negative], outputs = loss_layer)

# create simple callback for projecting embeddings after every epoch
projector = TBProjectorCallback(
    base_network,
    "tb",
    test_images,
    test_labels,
    batch_size=BATCH_SIZE,
    image_size=64,
    normalize_fn=normalize_images,
    normalize_eb=True,
    freq=1,
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
    freq=1,
)

triplet_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001))

triplet_model.fit(
    AnchorPositiveNegative(base_network, train_images, train_labels),
    callbacks=[evaluator, projector],
    epochs=EPOCHS
)

import tensorflow as tf
import numpy as np

import random
import cv2

from tf_metric_learning.miners.annoy import TripletAnnoyMiner


def store_images(filename, *images):
    new_images = [cv2.cvtColor(image * 255.0, cv2.COLOR_RGB2BGR) for image in images]

    if random.random() < 0.1:
        cv2.imwrite(filename, np.concatenate(new_images, axis=1))


class BaseMinerSequence(tf.keras.utils.Sequence):
    def __init__(self, base_model, train_images, train_labels, embedding_size, batch_size):
        self.base_model = base_model
        self.train_images = train_images
        self.train_labels = train_labels
        self.batch_size = batch_size

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
        return len(self.train_images) // self.batch_size

    def on_epoch_end(self):
        embeddings = self.base_model.predict(self.train_images)
        np.random.shuffle(self.indexes)
        self.epoch += 1
        if self.epoch >= 20:
            self.miner.reindex(embeddings)

    def pick_positive(self, anchor_image, anchor_label, id):
        """
        Current implementation is picking random positive sample from same class.
        You can reimplement/override this mechanism.
        """
        return random.choice(self.mapping[anchor_label])

    def pick_negative(self, anchor_image, anchor_label, id):
        """
        Pick negative sample.
        """
        if self.epoch < 20:
            # randomly pick easy negative image
            numbers = list(range(0, len(self.mapping.keys())))
            numbers.remove(anchor_label)
            random_label = random.choice(numbers)
            return random.choice(self.mapping[random_label])
        else:
            # in second phase, pick a bit harder sample
            return self.miner.search_easiest_negative(self.miner.get_item_vector(id), anchor_label, n=100)

    @tf.function
    def augment_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, 0.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image

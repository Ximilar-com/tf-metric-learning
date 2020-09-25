import tensorflow as tf
import tensorflow_addons as tfa

from tf_metric_learning.utils.constants import *


class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin=0.2, normalize=False, weight=1.0, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)

        self.margin = margin
        self.normalize = normalize
        self.weight = weight

    def get_config(self):
        config = {"margin": self.margin, "normalize": self.normalize, "weight": self.weight}
        base_config = super(TripletLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings_a, embeddings_p, embeddings_n):
        distance_pos = self.euclidean_distance(embeddings_a, embeddings_p)
        distance_neg = self.euclidean_distance(embeddings_a, embeddings_n)
        triplet_loss = tf.maximum(0.0, self.margin + distance_pos - distance_neg)
        total_loss = tf.reduce_sum(triplet_loss)
        return total_loss, distance_pos, distance_neg

    def euclidean_distance(self, a, b):
        return tf.maximum(0.0, tf.reduce_sum(tf.square(tf.subtract(a, b)), 1))

    def call(self, inputs):
        embeddings_a, embeddings_p, embeddings_n = inputs[ANCHOR], inputs[POSITIVE], inputs[NEGATIVE]

        if self.normalize:
            embeddings_a = tf.nn.l2_normalize(embeddings_a, axis=1)
            embeddings_p = tf.nn.l2_normalize(embeddings_p, axis=1)
            embeddings_n = tf.nn.l2_normalize(embeddings_n, axis=1)

        loss, distance_pos, distance_neg = self.loss_fn(embeddings_a, embeddings_p, embeddings_n)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        self.add_metric(distance_pos, name="distance_pos", aggregation="mean")
        self.add_metric(distance_neg, name="distance_neg", aggregation="mean")
        return inputs

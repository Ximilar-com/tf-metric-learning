import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

from tf_metric_learning.utils.constants import *


class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self, margin=1.0, normalize=True, crossentropy=False, weight=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)

        self.normalize = normalize
        self.margin = margin
        self.crossentropy = crossentropy
        self.weight = weight

    def get_config(self):
        config = {
            "normalize": self.normalize,
            "margin": self.margin,
            "crossentropy": self.crossentropy,
            "weight": self.weight,
        }
        base_config = super(ContrastiveLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings_1, embeddings_2, y_true):
        y_pred = tf.linalg.norm(embeddings_1 - embeddings_2, axis=1)

        if self.crossentropy:
            loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
            return tf.reduce_mean(loss)
        else:
            loss = tfa.losses.contrastive_loss(y_true, y_pred, margin=self.margin)
            return tf.reduce_mean(loss)

    def euclidean_distance(self, x, y):
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def call(self, inputs):
        embeddings_1, embeddings_2, labels = inputs[ANCHOR], inputs[POSITIVE], inputs[LABELS]

        if self.normalize:
            embeddings_1 = tf.nn.l2_normalize(embeddings_1, axis=1)
            embeddings_2 = tf.nn.l2_normalize(embeddings_2, axis=1)

        labels = tf.reshape(labels, [-1])

        # tfa.losses.contrastive_loss uses notion that 1.0 are positive pairs
        # and label 0 is negative pair, even though the l2 distance is inverse :)
        pos_idx = tf.reshape(tf.where(labels == 1.0), [-1])
        neg_idx = tf.reshape(tf.where(labels == 0.0), [-1])

        distance_pos = self.euclidean_distance(tf.gather(embeddings_1, pos_idx), tf.gather(embeddings_2, pos_idx))
        distance_neg = self.euclidean_distance(tf.gather(embeddings_1, neg_idx), tf.gather(embeddings_2, neg_idx))

        loss = self.loss_fn(embeddings_1, embeddings_2, labels)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        self.add_metric(distance_pos, name="distance_pos", aggregation="mean")
        self.add_metric(distance_neg, name="distance_neg", aggregation="mean")
        return inputs

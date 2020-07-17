import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K


class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin=0.5, normalize=True, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)

        self.margin = margin
        self.normalize = normalize

    def get_config(self):
        config = {
            "margin": self.margin,
            "normalize": self.normalize
        }
        base_config = super(TripletLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings_a, embeddings_p, embeddings_n):
        distance_pos = self.euclidean_distance(embeddings_a, embeddings_p)
        distance_neg = self.euclidean_distance(embeddings_a, embeddings_n)
        triplet_loss = tf.maximum(0.0, self.margin + distance_pos - distance_neg)
        total_loss = tf.reduce_mean(triplet_loss)
        return total_loss

    def euclidean_distance(self, a, b):
        if self.normalize:
            # use this when using l2 norm vectors
            return tf.maximum(0.0, tf.reduce_sum(tf.square(tf.subtract(a, b)), 1))
        # otherwise use uclidean distance
        return tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b)), 1) + tf.keras.backend.epsilon()), 0.0)

    def call(self, embeddings_a, embeddings_p, embeddings_n):
        if self.normalize:
            embeddings_a = tf.nn.l2_normalize(embeddings_a, axis=1)
            embeddings_p = tf.nn.l2_normalize(embeddings_p, axis=1)
            embeddings_n = tf.nn.l2_normalize(embeddings_n, axis=1)

        loss = self.loss_fn(embeddings_a, embeddings_p, embeddings_n)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return embeddings_a, embeddings_p, embeddings_n

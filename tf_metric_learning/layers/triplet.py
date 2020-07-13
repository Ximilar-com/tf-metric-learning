import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K


class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin=1.0, normalize=False, **kwargs):
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
        anch_pos = self.euclidean_distance(embeddings_a, embeddings_p)
        anch_neg = self.euclidean_distance(embeddings_a, embeddings_n)
        pos_neg = self.euclidean_distance(embeddings_p, embeddings_n)

        return K.mean(K.maximum(K.constant(0), K.square(anch_pos) - 0.5*(K.square(anch_neg)+K.square(pos_neg)) + self.margin))

    def euclidean_distance(self, x, y):
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def call(self, embeddings_a, embeddings_p, embeddings_n):
        if self.normalize:
            embeddings_a = tf.nn.l2_normalize(embeddings_a, axis=1)
            embeddings_p = tf.nn.l2_normalize(embeddings_p, axis=1)
            embeddings_n = tf.nn.l2_normalize(embeddings_n, axis=1)

        loss = self.loss_fn(embeddings_a, embeddings_p, embeddings_n)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return embeddings_a, embeddings_p, embeddings_n

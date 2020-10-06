import tensorflow as tf
import tensorflow_addons as tfa

from tf_metric_learning.utils.constants import *


class NPairLoss(tf.keras.layers.Layer):
    """
    NPairLoss layer using tf.addons.
    Each pair in the batch must have unique label, that is why the labels
    are created as tf.range(batch_size). Embeddings extracted for this
    layer should not be normalized and you should use small reg_lambda to 
    force the network to learn normalized embeddings.
    """

    def __init__(self, reg_lambda=0.0, weight=1.0, **kwargs):
        super(NPairLoss, self).__init__(**kwargs)

        self.reg_lambda = reg_lambda
        self.weight = weight

    def get_config(self):
        config = {"reg_lambda": self.reg_lambda, "weight": self.weight}
        base_config = super(NPairLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings_a, embeddings_p, labels):
        d_matrix = tf.matmul(embeddings_a, embeddings_p, transpose_a=False, transpose_b=True)
        return tfa.losses.npairs_loss(labels, d_matrix) + self.l2norm(embeddings_a, embeddings_p)

    def l2norm(self, embeddings_a, embeddings_p):
        "Regularize output of embeddings"
        reg_anchor = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(embeddings_a), 1))
        reg_positive = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(embeddings_p), 1))
        l2loss = tf.math.multiply(self.reg_lambda, reg_anchor + reg_positive)
        return l2loss

    def euclidean_distance(self, x, y):
        x = tf.keras.backend.l2_normalize(x, axis=1)
        y = tf.keras.backend.l2_normalize(y, axis=1)
        return tf.reduce_mean(tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1), tf.keras.backend.epsilon())))

    def call(self, inputs):
        embeddings_a, embeddings_p = inputs[ANCHOR], inputs[POSITIVE]

        # as the labels are unique in batch, we are not using inputs[LABELS]
        labels_new = tf.range(tf.shape(embeddings_a)[0])
        loss = self.loss_fn(embeddings_a, embeddings_p, labels_new)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")

        if self.reg_lambda:
            self.add_metric(self.l2norm(embeddings_a, embeddings_p), name="l2norm", aggregation="mean")

        self.add_metric(self.euclidean_distance(embeddings_a, embeddings_p), name="distance_pos", aggregation="mean")
        return inputs

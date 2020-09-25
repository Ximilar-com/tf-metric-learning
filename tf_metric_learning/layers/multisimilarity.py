import tensorflow as tf

from tf_metric_learning.utils.constants import *


class MultiSimilarityLoss(tf.keras.layers.Layer):
    """
    The original implementation was taken from: https://github.com/geonm/tf_ms_loss
    """

    def __init__(self, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, weight=1.0, **kwargs):
        super(MultiSimilarityLoss, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.eps = eps
        self.weight = weight

    def get_config(self):
        config = {"alpha": self.alpha, "beta": self.beta, "lamb": self.lamb, "eps": self.eps, "weight": self.weight}
        base_config = super(MultiSimilarityLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def loss_fn(self, embeddings_1, embeddings_2):
        batch_size = tf.shape(embeddings_2)[0]

        embeddings = tf.concat([embeddings_1, embeddings_2], axis=0)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        labels = tf.concat([tf.range(batch_size), tf.range(batch_size)], axis=0)
        labels = tf.reshape(labels, [-1, 1])

        adjacency = tf.equal(labels, tf.transpose(labels))
        adjacency_not = tf.logical_not(adjacency)

        mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size * 2, dtype=tf.float32)
        mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

        sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
        sim_mat = tf.maximum(sim_mat, 0.0)

        pos_mat = tf.multiply(sim_mat, mask_pos)
        neg_mat = tf.multiply(sim_mat, mask_neg)

        pos_exp = tf.exp(-self.alpha * (pos_mat - self.lamb))
        pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

        neg_exp = tf.exp(self.beta * (neg_mat - self.lamb))
        neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

        pos_term = tf.math.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / self.alpha
        neg_term = tf.math.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / self.beta

        loss = tf.reduce_mean(pos_term + neg_term)
        return loss

    def call(self, inputs):
        embeddings_a, embeddings_p = inputs[ANCHOR], inputs[POSITIVE]
        loss = self.loss_fn(embeddings_a, embeddings_p)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return inputs

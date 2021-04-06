import tensorflow as tf

from tf_metric_learning.utils.constants import *


class MultiSimilarityLoss(tf.keras.layers.Layer):
    """
    The original implementation was taken from: https://github.com/geonm/tf_ms_loss
    Use tf.keras.Input(shape=(1), name="input_labels") for input.
    """

    def __init__(self, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, weight=1.0, mining=False, **kwargs):
        super(MultiSimilarityLoss, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.eps = eps
        self.weight = weight
        self.mining = mining

    def get_config(self):
        config = {"alpha": self.alpha, "beta": self.beta, "lamb": self.lamb, "eps": self.eps, "weight": self.weight}
        base_config = super(MultiSimilarityLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def loss_fn(self, embeddings, labels):
        batch_size = tf.size(labels)
        adjacency = tf.equal(labels, tf.transpose(labels))
        adjacency_not = tf.logical_not(adjacency)
        mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
        mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

        sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
        sim_mat = tf.maximum(sim_mat, 0.0)

        pos_mat = tf.multiply(sim_mat, mask_pos)
        neg_mat = tf.multiply(sim_mat, mask_neg)

        if self.mining:
            max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
            tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
            min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

            max_val = tf.tile(max_val, [1, batch_size])
            min_val = tf.tile(min_val, [1, batch_size])

            mask_pos = tf.where(pos_mat < max_val + self.eps, mask_pos, tf.zeros_like(mask_pos))
            mask_neg = tf.where(neg_mat > min_val - self.eps, mask_neg, tf.zeros_like(mask_neg))

        pos_exp = tf.exp(-self.alpha * (pos_mat - self.lamb))
        pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

        neg_exp = tf.exp(self.beta * (neg_mat - self.lamb))
        neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

        pos_term = tf.math.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / self.alpha
        neg_term = tf.math.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / self.beta

        loss = tf.reduce_mean(pos_term + neg_term)
        return loss

    def call(self, inputs):
        embeddings, labels = inputs[ANCHOR], inputs[LABELS]
        loss = self.loss_fn(embeddings, labels)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return inputs

import tensorflow as tf

from tf_metric_learning.utils.constants import *


class ProxyAnchorLoss(tf.keras.layers.Layer):
    def __init__(self, num_class, embeddings_size, margin=0.1, alpha=32.0, weight=1.0, **kwargs):
        super(ProxyAnchorLoss, self).__init__(**kwargs)

        self.margin = margin
        self.alpha = alpha
        self.num_class = int(num_class)
        self.embeddings_size = embeddings_size
        self.weight = weight

    def build(self, input_shape):
        self.proxy = self.add_weight(
            name="proxy",
            shape=[self.num_class, self.embeddings_size],
            initializer="he_normal",
            trainable=True,
            dtype=tf.float32,
        )

    def get_config(self):
        config = {
            "num_class": self.num_class,
            "embeddings_size": self.embeddings_size,
            "margin": self.margin,
            "alpha": self.alpha,
            "weight": self.weight,
        }
        base_config = super(ProxyAnchorLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def loss_fn(self, embeddings, labels):
        embeddings_l2 = tf.nn.l2_normalize(embeddings, axis=1)
        proxy_l2 = tf.nn.l2_normalize(self.proxy, axis=1)

        labels = tf.reshape(labels, [-1])
        pos_target = tf.one_hot(tf.cast(labels, tf.int32), self.num_class, dtype=tf.float32)
        neg_target = 1.0 - pos_target

        sim_mat = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

        pos_exp = tf.exp(-self.alpha * (sim_mat - self.margin))
        neg_exp = tf.exp(self.alpha * (sim_mat + self.margin))

        P_sim_sum = tf.reduce_sum(pos_exp * pos_target, axis=0)
        N_sim_sum = tf.reduce_sum(neg_exp * neg_target, axis=0)

        num_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(pos_target, axis=0), dtype=tf.dtypes.float32)

        pos_term = tf.reduce_sum(tf.math.log(1.0 + P_sim_sum)) / num_valid_proxies
        neg_term = tf.reduce_sum(tf.math.log(1.0 + N_sim_sum)) / self.num_class
        loss = pos_term + neg_term
        return loss

    def call(self, inputs):
        embeddings, labels = inputs[EMBEDDINGS], inputs[LABELS]
        loss = self.loss_fn(embeddings, labels)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return inputs

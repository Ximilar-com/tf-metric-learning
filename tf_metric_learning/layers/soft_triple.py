import tensorflow as tf

from tf_metric_learning.utils.constants import *


class SoftTripleLoss(tf.keras.layers.Layer):
    def __init__(self, num_class, num_centers, embeddings_size, weight=1.0, **kwargs):
        super(SoftTripleLoss, self).__init__(**kwargs)

        self.num_class = num_class
        self.num_centers = num_centers
        self.embeddings_size = embeddings_size
        self.weight = weight

    def build(self, input_shape):
        self.large_centers = self.add_weight(
            name="large_centers",
            shape=[self.num_class * self.num_centers, self.embeddings_size],
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32,
        )

    def get_config(self):
        config = {
            "num_class": self.num_class,
            "num_centers": self.num_centers,
            "embeddings_size": self.embeddings_size,
            "weight": self.weight,
        }
        base_config = super(SoftTripleLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings, labels, p_lambda=20.0, p_tau=0.2, p_gamma=0.1, p_delta=0.01):
        batch_size = tf.shape(embeddings)[0]
        large_centers = tf.nn.l2_normalize(self.large_centers, axis=-1)
        embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
        large_logits = tf.matmul(embeddings, large_centers, transpose_b=True)

        rs_large_logits = tf.reshape(large_logits, [batch_size, self.num_centers, self.num_class], name="first_reshape")
        exp_rs_large_logits = tf.exp((1.0 / p_gamma) * rs_large_logits)
        sum_rs_large_logits = tf.reduce_sum(exp_rs_large_logits, axis=1, keepdims=True)
        coeff_large_logits = exp_rs_large_logits / sum_rs_large_logits

        rs_large_logits = tf.multiply(rs_large_logits, coeff_large_logits)
        logits = tf.reduce_sum(rs_large_logits, axis=1, keepdims=False)

        gt = tf.reshape(
            labels, [-1], name="second_reshape"
        )  # [[0], [7], [3], [22], [39], ...] -> [0, 7, 3, 22, 39, ...]
        gt_int = tf.cast(gt, tf.int32)
        labels_map = tf.one_hot(gt_int, depth=self.num_class, dtype=tf.float32)

        # subtract p_delta
        delta_map = p_delta * labels_map
        logits_delta = logits - delta_map
        scaled_logits_delta = p_lambda * (logits_delta)

        # get xentropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits_delta, labels=labels_map)
        loss = tf.reduce_mean(loss, name="loss_xentropy")
        return loss

    def regularization(self, p_lambda=20.0, p_tau=0.2, p_gamma=0.1, p_delta=0.01):
        large_centers = tf.nn.l2_normalize(self.large_centers, axis=-1)
        sim_large_centers = tf.abs(
            tf.matmul(large_centers, large_centers, transpose_b=True)
        )  # [num_class * num_centers, num_class * num_centers]

        dist_large_centers = tf.sqrt(tf.abs(2.0 - 2.0 * sim_large_centers) + 1e-10)
        checkerboard = tf.range(self.num_class, dtype=tf.int32)
        checkerboard = tf.one_hot(checkerboard, depth=self.num_class, dtype=tf.float32)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, self.num_centers, axis=0)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, self.num_centers, axis=1)

        dist_large_centers = tf.multiply(dist_large_centers, checkerboard)
        mask = tf.ones_like(dist_large_centers, dtype=tf.float32) - tf.eye(
            self.num_class * self.num_centers, dtype=tf.float32
        )
        dist_large_centers = p_tau * tf.multiply(dist_large_centers, mask)
        reg_numer = tf.reduce_sum(dist_large_centers) / 2.0
        reg_denumer = self.num_class * self.num_centers * (self.num_centers - 1.0)

        loss_reg = reg_numer / reg_denumer
        return loss_reg

    def call(self, inputs):
        embeddings, labels = inputs[EMBEDDINGS], inputs[LABELS]
        loss = self.loss_fn(embeddings, labels)
        loss = loss * self.weight

        reg_loss = self.regularization()
        self.add_loss(loss + reg_loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        self.add_metric(reg_loss, name=self.name + "_reg", aggregation="mean")
        return inputs

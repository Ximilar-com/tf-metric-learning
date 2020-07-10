import tensorflow as tf
import tensorflow_addons as tfa


class NPairLoss(tf.keras.layers.Layer):
    def __init__(self, reg_lambda=0.0, **kwargs):
        super(NPairLoss, self).__init__(**kwargs)

        self.reg_lambda = reg_lambda

    def get_config(self):
        config = {
            "reg_lambda": self.reg_lambda,
        }
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

    def call(self, embeddings_a, embeddings_p, labels):
        loss = self.loss_fn(embeddings_a, embeddings_p, labels)

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")

        if self.reg_lambda:
            self.add_metric(self.l2norm(embeddings_a, embeddings_p), name="l2norm", aggregation="mean")

        return embeddings_a, embeddings_p, labels

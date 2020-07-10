import tensorflow as tf


class ProxyAnchorLoss(tf.keras.layers.Layer):
    def __init__(self, num_class, embeddings_size, margin=0.1, alpha=32.0, **kwargs):
        super(ProxyAnchorLoss, self).__init__(**kwargs)

        self.margin = margin
        self.alpha = alpha
        self.num_class = num_class
        self.embeddings_size = embeddings_size

    def build(self, input_shape):
        self.proxy = self.add_weight(
            name="proxy",
            shape=[self.num_class, self.embeddings_size],
            initializer="random_normal", 
            trainable=True,
            dtype=tf.float32
        )

    def get_config(self):
        config = {
            "num_class": self.num_class,
            "embeddings_size": self.embeddings_size
            "margin": self.margin,
            "alpha": self.alpha
        }
        base_config = super(ProxyAnchorLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings, labels):
        embeddings_l2 = tf.nn.l2_normalize(embeddings, axis=1)
        proxy_l2 = tf.nn.l2_normalize(self.proxy, axis=1)

        pos_target = tf.one_hot(labels, self.num_class, dtype=tf.float32)
        neg_target = 1.0 - pos_target

        sim_mat = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

        pos_mat = tf.exp(-self.alpha * (sim_mat - self.margin)) * pos_target
        neg_mat = tf.exp(self.alpha * (sim_mat + self.margin)) * neg_target

        P_sim_sum = tf.reduce_sum(pos_mat, axis=0)
        N_sim_sum = tf.reduce_sum(neg_mat, axis=0)

        num_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(pos_target, axis=0), dtype=tf.dtypes.float32)

        pos_term = tf.reduce_sum(tf.math.log(1.0 + P_sim_sum)) / num_valid_proxies
        neg_term = tf.reduce_sum(tf.math.log(1.0 + N_sim_sum)) / self.num_class
        loss = pos_term + neg_term
        return loss

    def call(self, embeddings, labels):
        loss = self.loss_fn(embeddings, labels)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return embeddings, labels
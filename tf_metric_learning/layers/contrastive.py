import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K


class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self, margin=0.5, normalize=False, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)

        self.normalize = normalize
        self.margin = margin

    def get_config(self):
        config = {
            "normalize": self.normalize,
            "margin": self.margin
        }
        base_config = super(ContrastiveLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss_fn(self, embeddings_1, embeddings_2, y_true):
        y_pred = self.euclidean_distance(embeddings_1, embeddings_2)
        return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(self.margin - y_pred, 0)))

    def euclidean_distance(self, x, y):
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def call(self, embeddings_1, embeddings_2, labels):
        if self.normalize:
            embeddings_1 = tf.nn.l2_normalize(embeddings_1, axis=1)
            embeddings_2 = tf.nn.l2_normalize(embeddings_2, axis=1)

        loss = self.loss_fn(embeddings_1, embeddings_2, labels)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        return embeddings_a, embeddings_p, embeddings_n

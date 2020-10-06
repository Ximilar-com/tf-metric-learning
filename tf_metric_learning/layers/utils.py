import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

from tf_metric_learning.utils.constants import *


class PairReshapeLayer(tf.keras.layers.Layer):
    """
    Concat pairs to list and return this list with labels.
    """

    def __init__(self, concat_labels=False, **kwargs):
        super(PairReshapeLayer, self).__init__(**kwargs)

        self.concat_labels = concat_labels

    def get_config(self):
        config = {"concat_labels": self.concat_labels}
        base_config = super(PairReshapeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        labels = tf.concat([inputs[LABELS], inputs[LABELS]], axis=0) if self.concat_labels else  inputs[LABELS]

        return {
            EMBEDDINGS: tf.concat([inputs[ANCHOR], inputs[POSITIVE]], axis=0),
            LABELS: labels
        }


class TripletReshapeLayer(tf.keras.layers.Layer):
    """
    Concat triplets to list and return this list with labels.
    """

    def __init__(self, **kwargs):
        super(TripletReshapeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return {
            EMBEDDINGS: tf.concat([inputs[ANCHOR], inputs[POSITIVE], inputs[NEGATIVE]], axis=0),
            LABELS: inputs[LABELS],
        }


class BatchSizeLayer(tf.keras.layers.Layer):
    def __init__(self, input_name=LABELS, **kwargs):
        super(BatchSizeLayer, self).__init__(**kwargs)
        self.input_name = input_name

    def get_config(self):
        config = {"input_name": self.input_name}
        base_config = super(PairReshapeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        batch_size = tf.shape(inputs[self.input_name])[0]
        self.add_metric(tf.cast(batch_size, tf.float32), name="batch_size", aggregation="mean")
        return inputs


class PairNormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairReshapeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return {
            ANCHOR: tf.keras.backend.l2_normalize(inputs[ANCHOR], axis=1),
            POSITIVE: tf.keras.backend.l2_normalize(inputs[POSITIVE], axis=1),
            LABELS: inputs[LABELS],
        }


class TripletNormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairReshapeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return {
            ANCHOR: tf.keras.backend.l2_normalize(inputs[ANCHOR], axis=1),
            POSITIVE: tf.keras.backend.l2_normalize(inputs[POSITIVE], axis=1),
            NEGATIVE: tf.keras.backend.l2_normalize(inputs[NEGATIVE], axis=1),
            LABELS: labels,
        }

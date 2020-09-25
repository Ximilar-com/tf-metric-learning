import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

from tf_metric_learning.utils.constants import *


class ClassificationLoss(tf.keras.layers.Layer):
    "Categorization loss function. (Softmax)"

    def __init__(self, weight=1.0, **kwargs):
        super(ClassificationLoss, self).__init__(**kwargs)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.metric_fn = tf.keras.metrics.sparse_categorical_accuracy
        self.weight = weight

    def get_config(self):
        config = {"weight": self.weight}
        base_config = super(ClassificationLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """
        Compute the Categorical cross entropy
        :param y_pred: output from dense layer with softmax where shape is [batch_size, classes]
        :param y_true: list of labels with shape [batch_size]
        :return: output, labels
        """
        y_pred, y_true = inputs[PREDICTION], inputs[TARGET]
        loss = self.loss_fn(y_true, y_pred)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        self.add_metric(self.metric_fn(y_true, y_pred), name="accuracy", aggregation="mean")
        return inputs


class TaggingLoss(tf.keras.layers.Layer):
    "MultiLabel/Binary loss function. (Sigmoid)"

    def __init__(self, weight=1.0, **kwargs):
        super(ClassificationLoss, self).__init__(**kwargs)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.metric_fn = tf.keras.metrics.binary_accuracy
        self.weight = weight

    def get_config(self):
        config = {"weight": self.weight}
        base_config = super(ClassificationLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """
        Compute the Binary cross entropy
        :param y_pred: output from dense layer with sigmoid where shape is [batch_size, classes]
        :param y_true: list of labels [batch_size, classes]
        :return: output, labels
        """
        y_pred, y_true = inputs[PREDICTION], inputs[TARGET]
        loss = self.loss_fn(y_true, y_pred)
        loss = loss * self.weight

        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation="mean")
        self.add_metric(self.metric_fn(y_true, y_pred), name="accuracy", aggregation="mean")
        return inputs

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K


class PairReshapeLayer(tf.keras.layers.Layer):
    """
    Concat pairs to list and return this list with labels.
    """
    def __init__(self, **kwargs):
        super(PairReshapeLayer, self).__init__(**kwargs)

    def call(self, embeddings_1, embeddings_2, labels):
        return tf.concat([embeddings_1, embeddings_2], axis=0), tf.concat([labels, labels], axis=0)


class TripletReshapeLayer(tf.keras.layers.Layer):
    """
    Concat triplets to list and return this list with labels.
    """
    def __init__(self, **kwargs):
        super(TripletReshapeLayer, self).__init__(**kwargs)

    def call(self, embeddings_1, embeddings_2, embeddings_3, labels):
        return tf.concat([embeddings_1, embeddings_2, embeddings_3], axis=0), labels


class DropLabelsLayer(tf.keras.layers.Layer):
    """
    Drop labels layer and get just the output.
    """
    def __init__(self, **kwargs):
        super(TripletReshapeLayer, self).__init__(**kwargs)

    def call(self, output, labels):
        return output

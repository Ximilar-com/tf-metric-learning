# tf-metric-learning

## Overview

This repository contains a TensorFlow2+/tf.keras implementation some of the loss functions and miners. This repository was inspired by [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning).

All the loss functions are implemented as tf.keras.layers.Layer.

#### Open-source repos
This library contains code that has been adapted and modified from the following great open-source repos:

* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [geonm](https://github.com/geonm?tab=repositories)
* [nixingyang](https://github.com/nixingyang/Proxy-Anchor-Loss)

### Examples

```python
import tensorflow as tf
import numpy as np
from tf_metric_learning.layers.soft_triple import SoftTripleLoss

num_class, num_centers, embedding_size = 10, 2, 256

inputs = tf.keras.Input(shape=(embedding_size), name="embeddings")
input_label = tf.keras.layers.Input(shape=(1,), name="labels")
output_tensor = SoftTripleLoss(num_class, num_centers, embedding_size)(inputs, input_label)

model = tf.keras.Model(inputs=[inputs, input_label], outputs=output_tensor)
model.compile(optimizer="adam")

data = {"embeddings" : np.asarray([np.zeros(256) for i in range(1000)]), "labels": np.zeros(1000, dtype=np.float32)}
model.fit(data, None, epochs=10, batch_size=10)
```

### Implementations

#### Loss functions

* [MultiSimilarityLoss](https://arxiv.org/abs/1904.06627) [TODO]
* [ProxyAnchorLoss](https://arxiv.org/abs/2003.13911) [TODO]
* [SoftTripleLoss](https://arxiv.org/abs/1909.05235) ✅
* [NPairLoss](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf) ✅
* [TripletLoss](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf) [TODO]
* [ContrastiveLoss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) [TODO]

#### Miners

* MaximumLossMiner [TODO]
* HardTripletMiner [TODO]

#### Visualizations

* Tensorboard Projector Callback [TODO]

#### Examples

* Simplet NPair Training on CIFAR 10 dataset
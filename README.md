# tf-metric-learning

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0) [![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

## Overview

Minimalistic open-source library for metric learning written in TensorFlow2, Numpy and OpenCV(CV2). This repository contains a TensorFlow2+/tf.keras implementation some of the loss functions and miners. This repository was inspired by [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning).

All the loss functions are implemented as tf.keras.layers.Layer.

#### Open-source repos
This library contains code that has been adapted and modified from the following great open-source repos, without them this will be not possible (THANK YOU):

* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [geonm](https://github.com/geonm?tab=repositories)
* [nixingyang](https://github.com/nixingyang/Proxy-Anchor-Loss)

### Examples

```python
import tensorflow as tf
import numpy as np
from tf_metric_learning.layers import SoftTripleLoss

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
* [ProxyAnchorLoss](https://arxiv.org/abs/2003.13911) ✅
* [SoftTripleLoss](https://arxiv.org/abs/1909.05235) ✅
* [NPairLoss](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf) ✅
* [TripletLoss](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf) ✅
* [ContrastiveLoss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) ✅

#### Miners

* MaximumLossMiner [TODO]
* HardTripletMiner [TODO]

#### Evaluators

* AnnoyEvaluator Callback: for evaluation Recall@K, you will need to install Spotify [annoy](https://github.com/spotify/annoy) library.

```python
import tensorflow as tf
from tf_metric_learning.utils.recall import AnnoyEvaluatorCallback

evaluator = AnnoyEvaluatorCallback(
    base_model,
    "log_dir",
    {"images": [...], "labels": [...]}, # images to store in index
    {"images": [...], "labels": [...]}, # images to query
    emb_size=256,
    metric="euclidean"
)
```

#### Visualizations

* Tensorboard Projector Callback

```python
import tensorflow as tf
from tf_metric_learning.utils.projector import TBProjectorCallback

def normalize_images(images):
    return images/255.0

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
...

projector = TBProjectorCallback(
    base_model,
    "log_dir",
    test_images, # list of images
    np.squeeze(test_labels),
    normalize_eb=True,
    normalize_fn=normalize_images
)
```

#### Examples

* Simple SoftTriple Training on CIFAR 10 with embeddings projector (**[LINK](examples/cifar10.py)**)
* ProxyAnchor Loss on Cars196, using tf.data.Dataset and projector  (**[LINK](examples/cars196.py)**)
* NPair Loss with MaximumLossMiner [TODO]
* TripletTraining [TODO]
* ContrastiveLoss on MNIST [TODO]
* Simple Classification on Cars196 with Projector Visualization and Evaluator [TODO]
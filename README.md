# tf-metric-learning

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0) [![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

## Overview

**Minimalistic** open-source library for metric learning written in [TensorFlow2](https://github.com/tensorflow/tensorflow), TF-Addons, Numpy, OpenCV(CV2) and [Annoy](https://github.com/spotify/annoy). This repository contains a TensorFlow2+/tf.keras implementation some of the loss functions and miners. This repository was inspired by [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning).

### Installation

Prerequirements:

    pip install tensorflow
    pip install tensorflow-addons
    pip install annoy
    pip install opencv-contrib-python

This library:

    pip install tf-metric-learning

### Features

* All the loss functions are implemented as tf.keras.layers.Layer
* Callbacks for Computing Recall, Visualize Embeddings in TensorBoard Projector
* Simple Mining mechanism with Annoy
* Combine multiple loss functions/layers in one model

#### Open-source repos
This library contains code that has been adapted and modified from the following great open-source repos, without them this will be not possible (THANK YOU):

* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [geonm](https://github.com/geonm?tab=repositories)
* [nixingyang](https://github.com/nixingyang/Proxy-Anchor-Loss)
* [daigo0927](https://github.com/daigo0927/tf-simple-metric-learning)

### TODO

* Discriminative layer optimizer (different learning rates) for Loss with weights (Proxy, SoftTriple, ...) [TODO](https://github.com/tensorflow/addons/pull/969)
* Some Tests 😇
* Improve and add more minerss

### Examples

```python
import tensorflow as tf
import numpy as np

from tf_metric_learning.layers import SoftTripleLoss
from tf_metric_learning.utils.constants import EMBEDDINGS, LABELS

num_class, num_centers, embedding_size = 10, 2, 256

inputs = tf.keras.Input(shape=(embedding_size), name=EMBEDDINGS)
input_label = tf.keras.layers.Input(shape=(1,), name=LABELS)
output_tensor = SoftTripleLoss(num_class, num_centers, embedding_size)({EMBEDDINGS:inputs, LABELS:input_label})

model = tf.keras.Model(inputs=[inputs, input_label], outputs=output_tensor)
model.compile(optimizer="adam")

data = {EMBEDDINGS : np.asarray([np.zeros(256) for i in range(1000)]), LABELS: np.zeros(1000, dtype=np.float32)}
model.fit(data, None, epochs=10, batch_size=10)
```

More complex scenarios:

* [Complex example with NPair Loss + Multi Similarity + Classification and Mining](examples/npair.py)
* [SoftTriple Training on CIFAR 10](examples/softriple.py)
* [ProxyAnchor Loss using tf.data.Dataset](examples/proxyanchor.py)
* [Triplet Training with Mining](examples/triplet.py)
* [Contrastive Training](examples/contrastive.py)
* [Classification baseline](examples/classification.py)

### Features

#### Loss functions

* [MultiSimilarityLoss](https://arxiv.org/abs/1904.06627) ✅
* [ProxyAnchorLoss](https://arxiv.org/abs/2003.13911) ✅
* [SoftTripleLoss](https://arxiv.org/abs/1909.05235) ✅
* [NPairLoss](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf) ✅
* [TripletLoss](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf) ✅
* [ContrastiveLoss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) ✅

#### Miners

* MaximumLossMiner [TODO]
* TripletAnnoyMiner ✅

#### Evaluators

* AnnoyEvaluator Callback: for evaluation Recall@K, you will need to install Spotify [annoy](https://github.com/spotify/annoy) library.

```python
import tensorflow as tf
from tf_metric_learning.utils.recall import AnnoyEvaluatorCallback

evaluator = AnnoyEvaluatorCallback(
    base_network,
    {"images": test_images[:divide], "labels": test_labels[:divide]}, # images stored to index
    {"images": test_images[divide:], "labels": test_labels[divide:]}, # images to query
    normalize_fn=lambda images: images / 255.0,
    normalize_eb=True,
    eb_size=embedding_size,
    freq=1,
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
    "tb/projector",
    test_images, # list of images
    np.squeeze(test_labels),
    normalize_eb=True,
    normalize_fn=normalize_images
)
```

"""
Implementation of several metric learning operations in TensorFlow 2+.
"""

__version__ = "0.1.0"

from . import layers, miners

from pkg_resources import declare_namespace

declare_namespace("tf_metric_learning")

from setuptools import setup, find_packages

import os

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tf-metric-learning",
    version=get_version("tf_metric_learning/__init__.py"),
    description="Image similarity, metric learning loss functions for TensorFlow 2+.",
    url="https://github.com/Ximilar-com/tf-metric-learning",
    author="Michal Lukac & Ximilar.com Team",
    author_email="tech@ximilar.com",
    license="MIT",
    packages=find_packages(),
    keywords="machine learning, multimedia, image",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
    namespace_packages=["tf_metric_learning"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

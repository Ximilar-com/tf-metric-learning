import tensorflow as tf

import os
import numpy as np

from annoy import AnnoyIndex
from tqdm import tqdm


class AnnoyDataIndex(tf.keras.callbacks.Callback):
    def __init__(self, eb_size, labels, metric="euclidean", save_dir=None, progress=True, **kwargs):
        super().__init__(**kwargs)

        self.progress = progress
        self.index = None
        self.metric = metric
        self.eb_size = eb_size
        self.save_dir = save_dir
        self.labels = labels
        self.ids = self.create_ids(labels)

    def create_ids(self, labels):
        return {i: label for i, label in enumerate(labels)}

    def get_label(self, index):
        return self.ids[index]

    def create_index(self):
        self.index = AnnoyIndex(self.eb_size, self.metric)

    def build(self, k=5):
        self.index.build(k)

    def load_index_file(self, file_path):
        self.index = AnnoyIndex(self.eb_size, self.metric)
        self.index.load(file_path, prefault=False)

    def add_to_index(self, i, embedding):
        self.index.add_item(i, embedding)

    def reindex(self, embeddings):
        self.create_index()
        self.add_to_index(embeddings)
        self.build(10)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.index.save(os.path.join(self.save_dir, "index.ann"))

    def get_item_vector(self, id):
        return self.index.get_item_vector(id)

    def search(self, embedding, include_distances=False, n=20):
        return self.index.get_nns_by_vector(embedding, n, search_k=-1, include_distances=include_distances)

import tensorflow as tf

import os
import numpy as np

from annoy import AnnoyIndex
from tqdm import tqdm


class AnnoyDataIndex(tf.keras.callbacks.Callback):
    def __init__(self, emb_size, metric, labels, save_dir=False, progress=True, **kwargs):
        super().__init__(**kwargs)

        self.progress = progress
        self.index = None
        self.metric = metric
        self.emb_size = emb_size
        self.save_dir = save_dir
        self.labels = labels
        self.ids = self.create_ids(labels)

    def create_ids(self, labels):
        return {i:label for i, label in enumerate(labels)}

    def get_label(self, index):
        return self.ids[index]

    def load_index_file(self, file_path):
        self.index = AnnoyIndex(self.emb_size, self.metric)
        self.index.load(file_path, prefault=False)

    def reindex(self, embeddings):
        self.index = AnnoyIndex(self.emb_size, self.metric)

        for i, embedding in tqdm(enumerate(embeddings), ncols=100, total=len(embeddings), disable=not self.progress, desc="Indexing ... "):
            self.index.add_item(i, embedding)

        self.index.build(10)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.index.save(os.path.join(self.save_dir, "index.ann"))

    def search(self, embedding, include_distances=False, n=20):
        return self.index.get_nns_by_vector(embedding, n, search_k=-1, include_distances=include_distances)

    def mine(self, ids, distances, label, operator, index):
        labels = np.asarray([self.get_label(result) for result in ids])
        indexes = np.where(labels==label)[0] if operator else np.where(labels!=label)[0]
        item_id =  ids[indexes[index]] if len(indexes) else None
        return item_id

    def search_hardest_negative(self, embedding, label, n=20):
        ids, distances = self.search(embedding, include_distances=True, n=n)
        return self.mine(ids, distances, label, False, 0)

    def search_hardest_positive(self, embedding, label, n=20):
        results = self.search(embedding, include_distances=True, n=n)
        return self.mine(ids, distances, label, True, -1)

    def search_easiest_positive(self, embedding, label, n=20):
        results = self.search(embedding, include_distances=True, n=n)
        return self.mine(ids, distances, label, True, 0)

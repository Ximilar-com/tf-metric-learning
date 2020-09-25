import tensorflow as tf

import os
import numpy as np

from tf_metric_learning.utils.index import AnnoyDataIndex


class TripletAnnoyMiner(AnnoyDataIndex):
    def __init__(
        self,
        base_model,
        eb_size,
        labels,
        metric="euclidean",
        save_dir=None,
        progress=False,
        normalize_eb=True,
        normalize_fn=None,
        **kwargs
    ):
        super().__init__(eb_size, labels, metric=metric, save_dir=save_dir, progress=progress)

        self.base_model = base_model
        self.normalize_eb = normalize_eb
        self.normalize_fn = normalize_fn

    def extract_embeddings(self, data):
        data = self.normalize_fn(data) if self.normalize_fn is not None else data
        embeddings = self.base_model.predict(data)
        if self.normalize_eb:
            embeddings = tf.nn.l2_normalize(embeddings, axis=1).numpy()
        return embeddings

    def mine_item(self, ids, distances, label, operator, index):
        labels = np.asarray([self.get_label(result) for result in ids])
        indexes = np.where(labels == label)[0] if operator else np.where(labels != label)[0]
        item_id = ids[indexes[index]] if len(indexes) else None
        return item_id

    def search_hardest_negative(self, embedding, label, n=20):
        ids, distances = self.search(embedding, include_distances=True, n=n)
        return self.mine_item(ids, distances, label, False, 0)

    def search_easiest_negative(self, embedding, label, n=20):
        ids, distances = self.search(embedding, include_distances=True, n=n)
        return self.mine_item(ids, distances, label, False, -1)

    def search_hardest_positive(self, embedding, label, n=20):
        results = self.search(embedding, include_distances=True, n=n)
        return self.mine_item(ids, distances, label, True, -1)

    def search_easiest_positive(self, embedding, label, n=20):
        results = self.search(embedding, include_distances=True, n=n)
        return self.mine_item(ids, distances, label, True, 0)

    def search_hardest_negative_image(self, image, label, n=20):
        embedding = self.extract_embeddings(np.asarray([image]))[0]
        return self.search_hardest_negative(embedding, label, n=n)

    def search_easiest_negative_image(self, image, label, n=20):
        embedding = self.extract_embeddings(np.asarray([image]))[0]
        return self.search_easiest_negative(embedding, label, n=n)

    def search_hardest_positive_image(self, image, label, n=20):
        embedding = self.extract_embeddings(np.asarray([image]))[0]
        return self.search_hardest_positive(embedding, label, n=n)

    def search_easiest_positive_image(self, image, label, n=20):
        embedding = self.extract_embeddings(np.asarray([image]))[0]
        return self.search_easiest_positive(embedding, label, n=n)

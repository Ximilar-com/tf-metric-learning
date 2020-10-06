import tensorflow as tf

import numpy as np
from tqdm import tqdm

from tf_metric_learning.utils.index import AnnoyDataIndex


class AnnoyEvaluatorCallback(AnnoyDataIndex):
    """
    Callback, extracts embeddings, add them to AnnoyIndex and evaluate them as recall.
    """

    def __init__(
        self,
        model,
        data_store,
        data_search,
        save_dir=None,
        eb_size=256,
        metric="euclidean",
        freq=1,
        batch_size=None,
        normalize_eb=True,
        normalize_fn=None,
        progress=True,
        **kwargs
    ):
        super().__init__(eb_size, data_store["labels"], metric=metric, save_dir=save_dir, progress=progress)

        self.base_model = model
        self.data_store = data_store
        self.data_search = data_search
        self.batch_size = batch_size
        self.freq = int(freq)
        self.normalize_eb = normalize_eb
        self.normalize_fn = normalize_fn
        self.results = {}

    def on_epoch_begin(self, epoch, logs=None):
        if self.freq and epoch % self.freq == 0:
            store_images = (
                self.normalize_fn(self.data_store["images"])
                if self.normalize_fn is not None
                else self.data_store["images"]
            )
            search_images = (
                self.normalize_fn(self.data_search["images"])
                if self.normalize_fn is not None
                else self.data_search["images"]
            )

            embeddings_store = self.base_model.predict(store_images, batch_size=self.batch_size)
            embeddings_search = self.base_model.predict(search_images, batch_size=self.batch_size)

            if self.normalize_eb:
                embeddings_store = tf.nn.l2_normalize(embeddings_store, axis=1).numpy()
                embeddings_search = tf.nn.l2_normalize(embeddings_search, axis=1).numpy()

            self.reindex(embeddings_store)
            self.evaluate(embeddings_store, embeddings_search)

    def evaluate(self, embeddings_store, embeddings_search):
            self.results = {"default": []}
            for i, embedding in tqdm(
                enumerate(embeddings_search),
                ncols=100,
                total=len(embeddings_search),
                disable=not self.progress,
                desc="Search/Recall",
            ):
                annoy_results = self.search(embedding, n=20, include_distances=False)
                annoy_results = [self.get_label(result) for result in annoy_results]
                recalls = self.eval_recall(annoy_results, self.data_search["labels"][i], [1, 4, 10, 20])
                self.results["default"].append(recalls)

            print("\nRecall@[1, 3, 5, 10, 20] Computed:", np.mean(np.asarray(self.results["default"]), axis=0), "\n")

    def eval_recall(self, annoy_results, label, recalls):
        return [1 if label in annoy_results[:recall_n] else 0 for recall_n in recalls]

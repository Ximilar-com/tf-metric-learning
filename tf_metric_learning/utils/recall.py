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
            self.compute_data()

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def compute_data(self):
        self.create_index()
        with tqdm(total=len(self.data_store["images"]), desc="Indexing ... ") as pbar:
            for batch in self.batch(self.data_store["images"], n=self.batch_size*10):
                store_images = self.normalize_fn(batch) if self.normalize_fn is not None else batch
                embeddings_store = self.base_model.predict(store_images, batch_size=self.batch_size)
                if self.normalize_eb:
                    embeddings_store = tf.nn.l2_normalize(embeddings_store, axis=1).numpy()
                self.add_to_index(embeddings_store)
                pbar.update(len(batch))
        self.build(k=5)
        self.evaluate(self.data_search["images"])

    def evaluate(self, images):
        self.results = {"default": []}

        with tqdm(total=len(images), desc="Evaluating ... ") as pbar:
            for batch in self.batch(images, n=self.batch_size*10):
                search_images = self.normalize_fn(batch) if self.normalize_fn is not None else batch
                embeddings_search = self.base_model.predict(search_images, batch_size=self.batch_size)
                if self.normalize_eb:
                    embeddings_search = tf.nn.l2_normalize(embeddings_search, axis=1).numpy()
                for i, embedding in enumerate(embeddings_search):
                    annoy_results = self.search(embedding, n=20, include_distances=False)
                    annoy_results = [self.get_label(result) for result in annoy_results]
                    recalls = self.eval_recall(annoy_results, self.data_search["labels"][i], [1, 4, 10, 20])
                    self.results["default"].append(recalls)
                pbar.update(len(batch))

            print("\nRecall@[1, 3, 5, 10, 20] Computed:", np.mean(np.asarray(self.results["default"]), axis=0), "\n")

    def eval_recall(self, annoy_results, label, recalls):
        return [1 if label in annoy_results[:recall_n] else 0 for recall_n in recalls]

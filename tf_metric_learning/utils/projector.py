import tensorflow as tf
import cv2
import numpy as np
import os

from tensorboard.plugins import projector

EMBEDDINGS = "embeddings"
METAFILE = "meta.tsv"
SPRITESFILE = "sprites.png"


class TBProjectorCallback(tf.keras.callbacks.Callback):
    """
    Callback, extracts embeddings and add them to TensorBoard Projector.
    """

    def __init__(
        self,
        model,
        log_dir,
        data_images,
        data_labels,
        show_images=True,
        image_size=32,
        freq=1,
        batch_size=None,
        normalize_eb=True,
        normalize_fn=None,
        **kwargs
    ):
        """
        Initialize callback for visuallizing embeddings into tensorflow projector.

        ! Currently due to some bugs in tensorboard, you need to specify different log_dir as your
        ! dir for tf.keras.callbacks.TensorBoard

        :param model: base model, should output embeddings
        :param log_dir: path to the tensorboard directory
        :param data_images: list of images
        :param data_labels: list of labels
        :param show_images: if we want to visualize image as point in projector (True)
        :param image_size: sprite image size, defaults to 32
        :param freq: default 1 (create projector every epoch)
        :param batch_size: None
        :param normalize_eb: normalize embeddings before projection
        :param normalize_fn: function to normalize images for embeddings extraction
        """
        super().__init__(**kwargs)

        self.base_model = model
        self.image_size = image_size
        self.data_images = data_images
        self.data_labels = data_labels
        self.batch_size = batch_size
        self.freq = int(freq)
        self.log_dir = log_dir
        self.show_images = show_images
        self.normalize_eb = normalize_eb
        self.normalize_fn = normalize_fn

    def on_epoch_end(self, epoch, logs=None):
        if self.freq and epoch % self.freq == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            self.save_labels_tsv(self.data_labels)
            self.create_sprite(self.data_images)
            self.register_embedding()

            data = self.normalize_fn(self.data_images) if self.normalize_fn is not None else self.data_images
            embeddings = self.base_model.predict(data, batch_size=self.batch_size)

            if self.normalize_eb:
                embeddings = tf.nn.l2_normalize(embeddings, axis=1)

            tensor_embeddings = tf.Variable(embeddings, name=EMBEDDINGS)

            # this is not working right now https://github.com/tensorflow/tensorboard/issues/2471
            # checkpoint = tf.train.Checkpoint(embedding=tensor_embeddings)
            # checkpoint.save(os.path.join(self.log_dir, EMBEDDINGS + ".ckpt"))

            saver = tf.compat.v1.train.Saver([tensor_embeddings])
            saver.save(sess=None, global_step=epoch, save_path=os.path.join(self.log_dir, EMBEDDINGS + ".ckpt"))

    def save_labels_tsv(self, labels):
        if os.path.exists(os.path.join(self.log_dir, METAFILE)):
            return

        with open(os.path.join(self.log_dir, METAFILE), "w") as f:
            for label in labels:
                f.write("{}\n".format(str(label)))

    def create_sprite(self, images):
        if os.path.exists(os.path.join(self.log_dir, SPRITESFILE)) or not self.show_images:
            return

        data = np.asarray([cv2.resize(image, (self.image_size, self.image_size)) for image in images])

        # for black & white or greyscale images
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
        data = np.pad(data, padding, mode="constant", constant_values=0)

        # tile images into sprite
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
        sprite = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        sprite = cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.log_dir, SPRITESFILE), sprite)

    def register_embedding(self):
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = EMBEDDINGS
        embedding.metadata_path = METAFILE

        # this adds the sprite images
        if self.show_images:
            embedding.sprite.image_path = SPRITESFILE
            embedding.sprite.single_image_dim.extend((self.image_size, self.image_size))

        projector.visualize_embeddings(self.log_dir, config)

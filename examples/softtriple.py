import tensorflow as tf
import numpy as np
from tf_metric_learning.layers.soft_triple import SoftTripleLoss

input_shape = (32, 32, 3)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define base network for embeddings
inputs = tf.keras.Input(shape=input_shape, name="images")
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet", layers=tf.keras.layers)(inputs)
pool = tf.keras.layers.GlobalAveragePooling2D()(model)
dropout = tf.keras.layers.Dropout(0.5)(pool)
embeddings = tf.keras.layers.Dense(units = embedding_size)(dropout)
base_network = tf.keras.Model(inputs = inputs, outputs = embeddings)

# Define the input and output tensors
input_label = tf.keras.layers.Input(shape=(1,), name="labels")
output_tensor = SoftTripleLoss(num_class, num_centers, embedding_size)(base_network.outputs[0], input_label)

# Define the model and compile it
model = tf.keras.Model(inputs=[inputs, input_label], outputs=output_tensor)
model.compile(optimizer="adam")

data = {
    "images" : train_images,
    "labels": train_labels
}

model.fit(data, train_labels, epochs=10, batch_size=10)
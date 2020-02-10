from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager Mode: ", tf.executing_eagerly())
print("Hub Version: ", hub.__version__)
if tf.config.experimental.list_physical_devices("GPU"):
    print("GPU is available")
else:
    print("GPU is not available")

# Split the training set into 60% and 40% so 35,000 examples are used for
# training (15,000 training, 10,000 validation) and 25,000 are used for testing
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

# Load the training set of imdb-reviews and split it accordingly
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True
)


train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch

'''
The neural network is created by stacking layers
This requires three main architectural decisions:
 - How to represent the text?
 - How many layers to use in the model?
 - How many hidden units to use for each layer?
 
In this example, the input data consists of sentences. The labels
to predict are either 0 or 1.

One way to represent the text is to convert sentences into embedding vectors.
We can use a pre-trained text embedding as the first layer, which will
have three advantages
 - We don't have to worry about text preprocessing.
 - We can benefit from transfer learning(?*)
 - The embedding has a fixed size, so it's simpler to process.
'''

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(
    embedding,
    input_shape=[],
    dtype=tf.string,
    trainable=True
)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))   # Using

model.summary()

'''   LOSS FUNCTION AND OPTIMIZER
A model needs a loss function and an optimizer for training. Since this is a
binary classification problem and the model outputs a probability 
(a single-unit layer with a sigmoid activation), we'll use the binary_crossentropy
loss function.

This isn't the only choice for a loss function, you could, for instance, choose
mean_squared_error. But, generally, binary_crossentropy is better for dealing with
probabilities - it measures the 'distance' between probability distributions, or
in our case, between the ground-truth distribution and the predictions.

Later, when we are exploringf regression problems (say, to predict the price of a
house), we will see how to use another loss function called meaned square error.
'''
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

''' 
Train the model for 20 epochs in mini-batches of 512 samples. This is 20 
iterations over all samples in the x_train and y_train tensors. While training,
monitor the model's loss and accuracy on the 10,000 samples from the validation set.
'''
history = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs=20,
    validation_data=validation_data.batch(512),
    verbose=1
)

''' EVALUATE THE MODEL
This fairly naive approach achieves an accuracy of about 87%.
With more advanced apporaches, the model should get closer to 95%.
'''
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

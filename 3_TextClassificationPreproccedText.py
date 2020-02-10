from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt



# Download the IMDB dataset
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an 8k~ vocabulary
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset
    # (instead of a dictionary)
    as_supervised=True,
    # Also return the 'info' structure.
    with_info=True
)

# Save the encoder
encoder = info.features['text'].encoder

''' PREPARE THE DATA FOR TRAINING
The reviews are all different lengths, so use padded_batch to zero pad the
sequences while batching.

Each batch will have a shape of (batch_size, sequence_length) because the
padding is dynamic each batch will have a different length.
'''
BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    # https://github.com/tensorflow/datasets/issues/102
    .padded_batch(32, tf.compat.v1.data.get_output_shapes(train_data))
)
test_batches = (
    test_data
    .padded_batch(32, tf.compat.v1.data.get_output_shapes(train_data))
)




''' BUILD THE MODEL
The first layer is an Embedding layer. This layer takes the integer-encoded
vocabulary and looks up the embedding vector for each word index. These
vectors are learned as the model trains. The vectors add a dimension to the
output array. The resulting dimensions are (batch, sequence, embedding)

Next, a GlobalAveragePooling1D layer returns a fixed-length output vector
for each example by averaging over the sequence dimension. This allows the model
to handle input of variable length, in the simplest way possible.

This fixed-length output vector is piped through a fully-connected (Dense) layer
with 16 hidden units.

The last layer is densely connected with a single output node. Using the sigmoid
activation function, this value is a float between 0 and 1, representing a
probability, or confidence level.
'''
model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),  # Includes the dense layer?
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

''' HIDDEN UNITS
The above model has two intermediate or "hidden" layers, between the input and
output. The number of outputs (units, nodes, or neurons) is the dimension of the
representational space for the layer. In other words, the amount of freedom
the network is allowed when learning an internal representation.

If a model has more hidden units (a higher-dimensional representation space)
and/or more layers, then the network can learn more complex representations.
However, it makes the network more computationally expensive and may lead to
learning unwanted patterns - patterns that improve performance on training data
but not on the test data. This is called overfitting, and we'll explore it later.
'''


history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches,
    validation_steps=30
)

# Evaluation
loss, accuracy = model.evaluate(test_batches)
print("Loss: ", loss)
print("Accuracy, ", accuracy)

# Create a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()  # Displays entries, one for each monitored metric during training and validation<

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training Loss')
# "b" is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # Clear figure
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
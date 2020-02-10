from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Self-made helper libraries
from OurPlots import *

# ---------- Fetch data set and make sure it's correct ----------

# Data set of clothing articles
fashion_set = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_set.load_data()
'''
 train_images (60000, 28, 28)  Sixty thousand images of 28x28 pixels
 train_labels (60000)          Sixty thousand labels (0-9)
 test_images  (10000, 28, 28)  Ten thousand images of 28x28 pixels
 test_labels  (10000)          Ten thousand labels (0-9)
'''

# Which labels represent which articles of clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Scale values from 0-255 -> 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

# ---------- Build, Train, Evaluate, Predict ----------
# Build the model
print("Building Model")
model = keras.Sequential([
    # Transforms the format of the images from a two-dimensional array
    # (of 28x28 pixels) to a one-dimensional array (of 28*28=784 pixels)
    keras.layers.Flatten(input_shape=(28, 28)),
    # Densely/Fully connected neural layers with 128 nodes
    keras.layers.Dense(128, activation='relu'),
    # 10-node softmax layer that returns an array of 10 probability
    # scores that sum to 1. Each node contains a score that indicates
    # the probability that the current image belongs to one of the 10
    # classes.
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
print("Compiling model")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training Model")
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
print("Evaluating Accuracy")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Make predictions
# (10000, 10) Ten thousand test images with 10-sized arrays of
# confidence values of labels.
print("Making predictions")
predictions = model.predict(test_images)

# Which of the indices in the first image has the highest confidence value
highest_confidence_index = np.argmax(predictions[500])
print("\nNetwork thinks it's a " + class_names[highest_confidence_index])

# Plot image, prediction, and prediction values for a single index
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot Image,prediction,values for lots of indices
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


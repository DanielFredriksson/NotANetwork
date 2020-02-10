from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from OurTools import *
from OurPlots import *

''' Download And Prepare the CIFAR10 dataset.
The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each
class. The dataset is divided into 50,000 training images and 10,000 testing images. The
classes are mutually exclusive and there is no overlap between them.
'''
print("\nDownloading CIF10 dataset")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


''' Create the Convolutional Base
The 6 lines of code below define the convolutional base using a common pattern:
a stack of Conv2D and MaxPooling2D layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels),
ignoring the batch size. If you are new to these dimensions, color_channels refers to (R,G,B).
In this example, you will configure our CNN to process inputs of shape (32, 32, 3), which is
the format of CIFAR images. You can do this by passing the argument 'input_shape' to our 
first layer.
'''
model = models.Sequential()
model.add(layers.Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(32, 32, 3)
))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = s_fit(
    model,
    name="Convolutional_1",
    train_data=(train_images, train_labels),
    test_data=(test_images, test_labels),
    # retrain=True,
    max_epochs=10
)

plot_accuracy_val_accuracy(history)

# Load the MIT-BIH database
import wfdb

# Read sample
record = wfdb.rdsamp('PhysioBankData/100', sampto=3000)
# Read annotation
annotation = wfdb.rdann('PhysioBankData/100', 'atr', sampto=3000)
# Plot data
wfdb.plot_all_records(directory='PhysioBankData')
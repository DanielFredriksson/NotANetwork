from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_accuracy_val_accuracy(history):
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['val_accuracy'], label='Val_Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


def plot_image(i, predictions_array, true_labels, img):
    prediction_array, true_label, img = predictions_array, true_labels[i], img[i]
    # Plot settings
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # Which is the next highest
    confidence = prediction_array[predicted_label]
    confidence_percent = 100*confidence
    prediction_array[predicted_label] = 0
    next_highest_confidence_label = np.argmax(prediction_array)
    next_highest_confidence_percent = 100*prediction_array[next_highest_confidence_label]
    prediction_array[predicted_label] = confidence

    plt.xlabel("{} {:2.0f}% ({})\n {:2.0f}% {}".format(
            class_names[predicted_label],
            confidence_percent,
            class_names[true_label],
            next_highest_confidence_percent,
            class_names[next_highest_confidence_label]
        ),
        color=color
    )


def plot_value_array(i, predictions_array, true_labels):
    prediction_array, true_label = predictions_array, true_labels[i]
    # Plot settings
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


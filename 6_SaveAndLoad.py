from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from OurTools import *


''' SAVE AND LOAD MODELS
Model progress can be saved during - and after - training. This means a model can resume where it left
off and avoid long training times. Saving also means you can share your model and others can recreate
your work. When publishing research models and techniques, most machine learning practitioners share:
 - Code to create the model.
 - The trained weights, or parameters, for the model.

Sharing this data helps other understand how the model works and try it themselves with new data.
'''

''' OPTIONS
There are different ways to save TensorFlow models - depending on the API you're using. This guide uses
tf.keras, a high-level API to build and train models in TensorFlow. For other approaches, see the 
TensorFlow 'Save And Restore' guide or 'Saving in eager'
'''

# !pip install -q pyyaml h5py      # Required to save models in HDf54 format

print(tf.version.VERSION)

''' GET AN EXAMPLE DATA SET
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


#Purely daniel testing out filepath saving.
test_model = tf.keras.models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784, )),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
test_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_data = (train_images, train_labels)
test_data = (test_images, test_labels)

'''
print("\nEvaluating untrained")
loss, acc = test_model.evaluate(test_images, test_labels, verbose=2)
print("\nEvaluating loaded weights")
s_fit(test_model, "Olivermodel", train_data, test_data)
loss, acc = test_model.evaluate(test_images, test_labels, verbose=2)
s_fit(test_model, "Olivermodel", train_data, test_data)
'''

''' DEFINE A MODEL
'''
def create_model():
    model = tf.keras.models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784, )),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = create_model()
model.summary()


''' SAVE CHECKPOINTS DURING TRAINING
You can use a trained model without having to retrain it, or pick-up training where you left off -
in case the training process was interrupted. The 'tf.keras.callbacks.ModelCheckpoint' callback
allows to continually save the model both during and at the end of training.
'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# Train the model with the new callback
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook) are in
# place to discourage outdated usage, and can be ignored.

'''
Create a new, untrained model. When restoring a model from weights-only, you must have a model with
the same architecture as the original model. Since it's the same model architecture, you can share
weights despite that it's a different instance of the model.

Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained model will
perform at chance levels (~10% accuracy)
'''
# Design & Compile
untrained_model = create_model()

# Evaluate
loss, acc = untrained_model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

'''
Now load the weights from the check-point and re-evaluate.
'''
# Load weights
untrained_model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = untrained_model.evaluate(test_images, test_labels, verbose=2)
print("Restored Model, accuracy: {:5.2}%".format(100 * acc))

''' CHECKPOINT CALLBACK OPTIONS
The callback provides several options to provide unique names for checkpoints and adjust the 
checkpointing frequency.

Train a new model, and save uniquely named checkpoints once every five epochs.
'''
# Include the epoch in the file name (uses 'str.format')
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5
)

# Create a new model instance
model = create_model()

# Save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(
    train_images,
    train_labels,
    epochs=50,
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0
)

# To test, reset the model and load the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)




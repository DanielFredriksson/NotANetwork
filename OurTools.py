from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os
import __main__ as main
import pickle


saved_models_dir = "Saved_Models"
checkpoint_filetype = '.index'
weights_filetype = '.ckpt'


# 's_fit' loads weights if the training has already been done, otherwise it trains normally.
def s_fit(model, name, train_data, test_data, retrain=False, max_epochs=20):
    file_dir = os.path.basename(main.__file__)[:-3]             # ..FileName..
    file_path = saved_models_dir + '/' + file_dir + '/' + name  # Saved_Models/FileName/name
    history_path = file_path + '_history'                       # Saved_Models/Filename/name_history

    train_data, train_validation = train_data
    test_data, test_validation = test_data

    if os.path.exists(file_path + checkpoint_filetype) and not retrain:
        print('\n' + name + " loaded earlier weights + history instead of training. (" + file_path + ")")
        path = file_path
        # Load Weights
        model.load_weights(path)
        # Load History
        history = pickle.load(open(history_path, 'rb'))

    else:
        # Define checkpoint which saves weights
        save_weights_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=file_path,
            save_weights_only=True,
            verbose=1
        )

        # Train the model
        print('\nModel \"' + name + "\" started training.")
        history = model.fit(
            train_data,
            train_validation,
            # steps_per_epoch=Something?,
            epochs=max_epochs,
            validation_data=(test_data, test_validation),
            callbacks=[save_weights_callback]   # Save Weights
        )
        history = history.history

        # Save history
        print("\nSaved weights + history to " + file_path)
        pickle.dump(history, open(history_path, 'wb'))

    return history



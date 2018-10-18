'''
Classification of Fashion MNIST images using a convolutional model written in Keras.
This example is using the Fashion MNIST database of clothing images provided by Zalando Research.
https://github.com/zalandoresearch/fashion-mnist

Author: IBM Watson
'''

import argparse
import gzip
import keras
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
import os
from os import environ
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sys
import time

# helper for tracking metrics. We will extend this class to HPOMetrics class in our script
from emetrics import EMetrics


###############################################################################
# Set up working directories for data, model and logs.
###############################################################################

model_filename = "mnist_cnn.h5"

# writing the train model and getting input data
if environ.get('RESULT_DIR') is not None:
    output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
    output_model_path = os.path.join(output_model_folder, model_filename)
else:
    output_model_folder = "model"
    output_model_path = os.path.join("model", model_filename)

os.makedirs(output_model_folder, exist_ok=True)

#writing metrics
if environ.get('JOB_STATE_DIR') is not None:
    tb_directory = os.path.join(os.environ["JOB_STATE_DIR"], "logs", "tb", "test")
else:
    tb_directory = os.path.join("logs", "tb", "test")

os.makedirs(tb_directory, exist_ok=True)
tensorboard = TensorBoard(log_dir=tb_directory)

###############################################################################





###############################################################################
# START Set up HPO
###############################################################################

# The config file contains information for the hyperparameters
# config.json is generaed by the HPO service -- the parameters come from the manifest file
config_file = "config.json"

print("IMPORTED OK")

# checking that the config file is there
if os.path.exists(config_file):
  # open it and fetch the params
    with open("config.json", 'r') as f:
        print("FOUND CONFIG...")
        json_obj = json.load(f)
    learning_rate = json_obj["learning_rate"]
    batch_size = json_obj["batch_size"]
    conv_filter_1 = json_obj["conv_filter_1"]
    dropout = json_obj["dropout"]

    print("learning_rate: ", learning_rate)
    print("conv_filter_1: ", conv_filter_1)
    print("batch_size: ", batch_size)
    print("dropout: ", dropout)

print("HYPERS: ", dropout, learning_rate, conv_filter_1, batch_size)

# the subID is how WML tracks the HPO step
# SUBID is an environment variable set by WML to the HPO iteration number 0,1,2,... the logs written by each iteration need to be kept separate
def getCurrentSubID():
    if "SUBID" in os.environ:
        return os.environ["SUBID"]
    else:
        return None

class HPOMetrics(keras.callbacks.Callback):
    def __init__(self):
        self.emetrics = EMetrics.open(getCurrentSubID())

    def on_epoch_end(self, epoch, logs={}):
        train_results = {}
        test_results = {}

        # this is where we check for the validation accuracy or those other metrics supplied by your model
        for key, value in logs.items():
            if 'val_' in key:
              # CHANGING THIS
                # key = key.split("_")[1]
                test_results.update({'accuracy': value})
            else:
                train_results.update({key: value})

        print('EPOCH ' + str(epoch))
        self.emetrics.record("train", epoch, train_results)
        self.emetrics.record(EMetrics.TEST_GROUP, epoch, test_results)

    def close(self):
        self.emetrics.close()

###############################################################################
# END Set up HPO
###############################################################################

print("HPO Config Complete")

# Add data dir to file path
data_dir = os.environ["DATA_DIR"]
train_images_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
test_images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
test_labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')


# Load data in MNIST format
with gzip.open(train_labels_file, 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), dtype=np.uint8,
                           offset=8)

with gzip.open(train_images_file, 'rb') as imgpath:
    x_train = np.frombuffer(imgpath.read(), dtype=np.uint8,
                           offset=16).reshape(len(y_train), 784)

with gzip.open(test_labels_file, 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), dtype=np.uint8,
                           offset=8)

with gzip.open(test_images_file, 'rb') as imgpath:
    x_test = np.frombuffer(imgpath.read(), dtype=np.uint8,
                           offset=16).reshape(len(y_test), 784)



print("Data loaded ...")

# Split a validation set off the train set
split = int(len(y_train) * .9)-1
x_train, x_val = x_train[:split], x_train[split:]
y_train, y_val = y_train[:split], y_train[split:]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Reshape to correct format for conv2d input
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


# timing
start_time = time.time()
print("Start time is {}".format(start_time))

# num_classes
num_classes=10


# epochs
epochs = 25

# build the model -- convolutional layers, pooling, dropout, fully connected layer
# notice that the hyperparameter values from config.json supplied to the model here 

print("Building model...")
model = Sequential()
model.add(Conv2D(conv_filter_1, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation='softmax'))


# The optimizer with learning rate from HPO
ada = keras.optimizers.Adadelta(lr = learning_rate)

# comple the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=ada,
              metrics=['accuracy'])


# dump to logs
model.summary()

# HPO tracking
hpo = HPOMetrics()



# tensorboard
tb_directory = os.environ["JOB_STATE_DIR"]+"/logs/tb/test"
tensorboard = TensorBoard(log_dir=tb_directory)

# fit the model
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard, hpo]) # ADD THIS CALLBACK HERE :)

# close HPO tracking
hpo.close()

print("Iteration complete")
print("Training history:" + str(history.history))

# evaluate on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print("Score: {}".format(score))

end_time = time.time()
minutes, seconds = divmod(end_time-start_time, 60)
print("Total train time: {:0>2}:{:05.2f}".format(int(minutes),seconds))

print('Final train accuracy:      %.4f' % history.history['acc'][-1])
print('Final train loss: %.4f' % history.history['loss'][-1])
print('Final validation accuracy: %.4f' % history.history['val_acc'][-1])
print('Final validation loss: %.4f' % history.history['val_loss'][-1])
print('Final test accuracy:       %.4f' %  score[1])
print('Final test loss: %.4f' % score[0])


# save the model
print("\nSaving model to: %s" % output_model_path)
model.save(output_model_path)


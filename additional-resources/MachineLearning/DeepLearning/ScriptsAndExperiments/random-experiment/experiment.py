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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import time

# Explore a random set of model hyperparameters.  This has been shown to be more
# efficient than a more limited grid search or manual hand tuning.
from random_search import random_search
def get_random_hyperparameters():

    params_search = random_search()
    params_search.add_static_var("batch_size",128)

    # Fashion MNIST converges around 25 epochs but
    params_search.add_static_var("epochs",25)

    params_search.add_power_range("num_filters_1",5,8,2) # 32 64 128 256
    params_search.add_power_range("num_filters_2",2,8,2) # 4 8 16 32 64 128 256
    params_search.add_power_range("num_filters_3",2,8,2) # 4 8 16 32 64 128 256
    params_search.add_step_range("pool_size_1",2,3,1)
    params_search.add_step_range("pool_size_2",2,3,1)
    params_search.add_step_range("filter_size_1",2,3,1)
    params_search.add_step_range("filter_size_2",2,3,1)
    params_search.add_step_range("filter_size_3",2,3,1)
    params_search.add_step_range("dropout_rate_1",0.1,0.9,0.1)
    params_search.add_step_range("dropout_rate_2",0.1,0.9,0.1)
    params_search.add_step_range("dropout_rate_3",0.1,0.9,0.1)
    params_search.add_step_range("dropout_rate_4",0.1,0.9,0.1)
    params_search.add_power_range("dense_neurons_1",2,12,2) # 4 8 16 32 64 128 256 512 1024 2048 4096 8192

    return params_search.get_random_hyperparameters()

params = get_random_hyperparameters()
print(params)

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

model = Sequential()
model.add(Conv2D(params["num_filters_1"], (params["filter_size_1"], params["filter_size_1"]),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(params["pool_size_1"], params["pool_size_1"])))
model.add(Dropout(params["dropout_rate_1"]))
model.add(Conv2D(params["num_filters_2"], (params["filter_size_2"], params["filter_size_2"]),
                 activation='relu',
                 kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(params["pool_size_2"], params["pool_size_2"])))
model.add(Dropout(params["dropout_rate_2"]))

model.add(Conv2D(params["num_filters_3"], (params["filter_size_3"], params["filter_size_3"]),
                    activation='relu',
                    kernel_initializer='he_normal'))
model.add(Dropout(params["dropout_rate_3"]))
model.add(Flatten())
model.add(Dense(params["dense_neurons_1"],activation='relu'))
model.add(Dropout(params["dropout_rate_4"]))

num_classes = 10
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

start_time = time.time()

tb_directory = os.environ["JOB_STATE_DIR"]+"/logs/tb/test"
tensorboard = TensorBoard(log_dir=tb_directory)
history = model.fit(x_train, y_train,
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, verbose=0)

end_time = time.time()
minutes, seconds = divmod(end_time-start_time, 60)
print("Total train time: {:0>2}:{:05.2f}".format(int(minutes),seconds))

print('Final train accuracy:      %.4f' % history.history['acc'][-1])
print('Final train loss: %.4f' % history.history['loss'][-1])
print('Final validation accuracy: %.4f' % history.history['val_acc'][-1])
print('Final validation loss: %.4f' % history.history['val_loss'][-1])
print('Final test accuracy:       %.4f' %  score[1])
print('Final test loss: %.4f' % score[0])

model_path = os.path.join(os.environ["RESULT_DIR"], "model.h5")
print("\nSaving model to: %s" % model_path)
model.save(model_path)

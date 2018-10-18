'''
    IBM Deep Learning (IDE) Generated Code.
    Compatible Keras Version : 2.1
    Tested on Python Version : 3.6.3
'''

# Choose the underlying compiler - tensorflow or theano
import json
import os 

with open(os.path.expanduser('~') + "/.keras/keras.json","r") as f:
    compiler_data = json.load(f)
compiler_data["backend"] = "tensorflow"
compiler_data["image_data_format"] = "channels_last"  
with open(os.path.expanduser('~') + '/.keras/keras.json', 'w') as outfile:
    json.dump(compiler_data, outfile)

# Global variable intilization
defined_metrics = []
defined_loss = ""

# import all the required packages
import numpy as np

import keras
from keras.models import Model
import keras.backend as K
import keras.regularizers as R
import keras.constraints as C
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Convolution2D, Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.optimizers import Adam



# Load data from pickle object

import pickle
class_labels_count = 1
with open('fashion_mnist-train.pkl', 'rb') as f:
    (train_data, train_label) = pickle.load(f)
    if (len(train_data.shape) == 3): 
        if('tensorflow' == 'tensorflow'):
            train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1).astype('float32') / 255   
        else:
            train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]).astype('float32') / 255   
    if (len(train_label.shape) == 1) or (len(train_label.shape) == 2 and train_label.shape[1] == 1):
        from keras.utils import np_utils
        class_labels_count = len(set(train_label.flatten()))
        train_label = np_utils.to_categorical(train_label, class_labels_count)
    else:
        class_labels_count = train_label.shape[1]

val_data = []
if('fashion_mnist-valid.pkl'):
    with open('fashion_mnist-valid.pkl', 'rb') as f:
        (val_data, val_label) = pickle.load(f)
        if (len(val_data.shape) == 3):
            if('tensorflow' == 'tensorflow'):
                val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], 1).astype('float32') / 255
            else:
                val_data = val_data.reshape(val_data.shape[0], 1, val_data.shape[1], val_data.shape[2]).astype('float32') / 255
        if (len(val_label.shape) == 1) or (len(val_label.shape) == 2 and val_label.shape[1] == 1):
            from keras.utils import np_utils
            val_label = np_utils.to_categorical(val_label, class_labels_count)
else:
    print('Validation set details not provided')
  
test_data = []
if('fashion_mnist-test.pkl'):
    with open('fashion_mnist-test.pkl', 'rb') as f:
        (test_data, test_label) = pickle.load(f)
        if (len(test_data.shape) == 3): 
            if('tensorflow' == 'tensorflow'):
                test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1).astype('float32') / 255
            else:
                test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2]).astype('float32') / 255
        if (len(test_label.shape) == 1) or (len(test_label.shape) == 2 and test_label.shape[1] == 1):
            from keras.utils import np_utils
            test_label = np_utils.to_categorical(test_label, class_labels_count)
else:
    print('Test set details not provided')

print(train_data.shape)
batch_input_shape_ImageData_a7d74d30 = train_data.shape[1:]
train_batch_size = 256

if True:

    #Input Layer
    ImageData_a7d74d30 = Input(shape=batch_input_shape_ImageData_a7d74d30)
    #Convolution2D Layer
    Convolution2D_1 = Convolution2D(32, (3, 3), kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal', padding = 'valid', strides = (1, 1), data_format = 'channels_last', use_bias = False, name = 'Convolution2D_2a2dd828')(ImageData_a7d74d30)
    #Batch Normalization Layer
    Convolution2D_1 = BatchNormalization(axis=3,name='bn_Convolution2D_2a2dd828')(Convolution2D_1)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_2 = Activation('relu', name = 'ReLU_08ca424c')(Convolution2D_1)
    #Pooling2D Layer
    Pooling2D_3 = MaxPooling2D(pool_size = (2, 2), padding = 'valid', data_format = 'channels_last', strides = (2, 2), name = 'Pooling2D_2fad9093')(ReLU_2)
    #Convolution2D Layer
    Convolution2D_4 = Convolution2D(64, (3, 3), kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal', padding = 'valid', strides = (1, 1), data_format = 'channels_last', use_bias = False, name = 'Convolution2D_416ca4b2')(Pooling2D_3)
    #Batch Normalization Layer
    Convolution2D_4 = BatchNormalization(axis=3,name='bn_Convolution2D_416ca4b2')(Convolution2D_4)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_5 = Activation('relu', name = 'ReLU_09f975fd')(Convolution2D_4)
    #Pooling2D Layer
    Pooling2D_6 = MaxPooling2D(pool_size = (2, 2), padding = 'valid', data_format = 'channels_last', strides = (2, 2), name = 'Pooling2D_1f8b9f47')(ReLU_5)
    #Convolution2D Layer
    Convolution2D_7 = Convolution2D(64, (3, 3), kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal', padding = 'valid', strides = (1, 1), data_format = 'channels_last', use_bias = False, name = 'Convolution2D_e2139fa8')(Pooling2D_6)
    #Batch Normalization Layer
    Convolution2D_7 = BatchNormalization(axis=3,name='bn_Convolution2D_e2139fa8')(Convolution2D_7)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_8 = Activation('relu', name = 'ReLU_f11d50bd')(Convolution2D_7)
    #Pooling2D Layer
    Pooling2D_9 = MaxPooling2D(pool_size = (2, 2), padding = 'valid', data_format = 'channels_last', strides = (2, 2), name = 'Pooling2D_4d87411b')(ReLU_8)
    #Flatten Layer
    Flatten_10 = Flatten(name = 'Flatten_8ed4f9b7')(Pooling2D_9)
    #Dense or Fully Connected (FC) Layer
    Dense_11 = Dense(10, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal', use_bias = False, name = 'Dense_59d0b43b')(Flatten_10)
    #Softmax Activation Layer
    Softmax_12 = Activation('softmax', name = 'Softmax_4bcd8d04')(Dense_11)
    #Accuracy Metric
    defined_metrics = ['accuracy']
    #SigmoidCrossEntropy Loss
    defined_loss = 'categorical_crossentropy'

    # Define a keras model
    model_inputs = [ImageData_a7d74d30]
    model_outputs = [Softmax_12]
    model = Model(inputs=model_inputs, outputs=model_outputs)

    # Set the required hyperparameters    
    num_epochs = 20

    # Defining the optimizer function
    adam_learning_rate = 0.1
    adam_decay = 0.1
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    optimizer_fn = Adam(lr=adam_learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2, decay=adam_decay)

    # performing final checks
    if not defined_metrics:
        defined_metrics=None
    if not defined_loss:
        defined_loss = 'categorical_crossentropy'
    if "ImageData" == "TextData" and "" == "Lang_Model":
        # adding a final Dense layer which has (vocab_length+1) units
        layers = [l for l in model.layers]
        for i in range(len(layers)):
            if isinstance(layers[i], keras.layers.core.Dense) and isinstance(layers[i+1], keras.layers.core.Activation):
                d = Dense(vocab_length+1, name = 'Dense_for_LM_' + str(i+1))(layers[i].output)
                layers[i+1].inbound_nodes = []              # assumption: there are no merges here
                d = layers[i+1](d)
        model = Model(inputs=layers[0].input, outputs=layers[len(layers)-1].output)
    
    # Compile and train the model
    model.compile(loss=defined_loss, optimizer=optimizer_fn, metrics=defined_metrics)
    
    if len(model_outputs) > 1: 
        train_label = [train_label] * len(model_outputs)
        if len(val_data) > 0: val_label = [val_label] * len(model_outputs)
        if len(test_data) > 0: test_label = [test_label] * len(model_outputs)
    
    # validate the model
    if (len(val_data) > 0):
        model.fit(train_data, train_label, batch_size=train_batch_size, epochs=num_epochs, verbose=1, validation_data=(val_data, val_label), shuffle=True)
    else:
        model.fit(train_data, train_label, batch_size=train_batch_size, epochs=num_epochs, verbose=1, shuffle=True)

    # test the model
    if (len(test_data) > 0):
        test_scores = model.evaluate(test_data, test_label, verbose=1)
        print(test_scores)

    # saving the model
    print('Saving the model...')
    if 'model_result_path' not in locals() and 'model_result_path' not in globals():
        model_result_path = "./keras_model.hdf5"
    model.save(model_result_path)
    print("Model saved in file: %s" % model_result_path)


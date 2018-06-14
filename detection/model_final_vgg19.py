#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import processing and generating functions
import generator_final as generator

# Import all libraries
import os
import json
from keras.models import Model
from keras.layers import Input, core, concatenate, Dense, MaxPooling2D, UpSampling2D, Dropout, Activation, Flatten, Conv2D, ELU, Lambda, SpatialDropout2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.regularizers import l1, l2

import keras.backend as K

import pandas as pd
import numpy as np

K.clear_session()

def set_samples_per_epoch(data_size, batch_size):
    
    # calculate max number of batches    
    num_batches = np.ceil(data_size/ batch_size)
    
    # calculate total number of sample comprised in epoch    
    samples_per_epoch = num_batches * batch_size
    
    return samples_per_epoch

def set_steps_per_epoch(data_size, batch_size):
    
    # calculate max number of batches    
    num_batches = np.ceil(data_size/ batch_size)
    
    return num_batches

test_name = "final"

# resizing images - needs to be a divisor of nb of conv layers for both values

ch, col, row = 3, 304, 144  # camera format
#ch, col, row = 3, 400, 192  # camera format

# Choose model
use_model = "FCN_Vgg19_unet"

smooth = 1.

#read file names from csv file
data_road = pd.read_csv('./data_road/lyft_road_data.csv', ";")
data_car = pd.read_csv('./data_car/lyft_car_data.csv', ";")

# First lets shuffle the dataset
data_nb_road = len(data_road)
print("Read in data_road, now have {} frames".format(data_nb_road))

data_nb_car = len(data_car)
print("Read in data_car, now have {} frames".format(data_nb_car))


# Shuffle to reduce time series issues
data_car.iloc[np.random.permutation(len(data_car))]
data_car.reset_index(drop=True)

data_road.iloc[np.random.permutation(len(data_road))]
data_road.reset_index(drop=True)

data_train_road = data_road
data_val_road = data_road
data_test_road = data_road

data_train_car = data_car
data_val_car = data_car
data_test_car = data_car

data_size_train_road = len(data_train_road)
data_size_val_road = len(data_val_road)
data_size_test_road = len(data_test_road)

data_size_train_car = len(data_train_car)
data_size_val_car = len(data_val_car)
data_size_test_car = len(data_test_car)

print("data_train_road has {} elements.".format(len(data_train_road)))
print("data_val_road has {} elements.".format(len(data_val_road)))
print("data_test_road has {} elements.".format(len(data_test_road)))

print("data_train_road has {} elements.".format(len(data_train_road)))
print("data_val_road has {} elements.".format(len(data_val_road)))
print("data_test_road has {} elements.".format(len(data_test_road)))

learning_rate = 1e-5

# Train model for n epochs and a batch size of x
epochs = 30 #150 for unet?
batch_size = 1

# create generators 
train_data_gen = generator.generate_batch(data_train_road, data_train_car, batch_size, ch, row, col)
val_data_gen = generator.generate_batch(data_val_road, data_val_car, batch_size, ch, row, col)

#test generator
batch_img, batch_mask = next(train_data_gen)

print("batch_img shape", batch_img.shape)
print("batch_img min", np.min(batch_img))
print("batch_img max", np.max(batch_img))
print("batch_img dtype", batch_img[0].dtype)

print("batch_mask shape", batch_mask.shape)
print("batch_mask min", np.min(batch_mask))
print("batch_mask max", np.max(batch_mask))
print("batch_mask dtype",batch_mask[0].dtype)

#show batch
"""
for i in range(batch_size):
    im = np.array(batch_img[i],dtype=np.uint8)
    im_mask = np.array(batch_mask[i],dtype=np.uint8)
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(im_mask[:,:,0], cmap='gray')
    #plt.imshow(im_mask[:,:,0]*255)
    plt.axis('off')
    plt.show();
"""
           
#Define IOU coefficient as loss function for a binary image
def IOU_calc(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)

def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

#average IOU loss for multiple binary images - channel 0:road - channel 1:cars
def IOU_multi_loss(y_true, y_pred):
    iou = 0.
    iou -= IOU_calc(y_true[:,:,:,0], y_pred[:,:,:,0])
    iou -= 9*IOU_calc(y_true[:,:,:,1], y_pred[:,:,:,1])
    return (iou/10.)

#Define network - UNET-FCN with VGG19 Layers and pre-trained weights
def FCN_Vgg19_unet(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=1):

    # Image Input
    ImageInput = Input(shape=(row, col, ch), name = 'image')

    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(ImageInput)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_regularizer=l2(weight_decay))(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', kernel_regularizer=l2(weight_decay))(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', kernel_regularizer=l2(weight_decay))(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv4')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv1')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv2')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv3')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block7_conv4')(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv1')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block8_conv2')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv')(conv9)
    
    output = Conv2D(2, (1, 1), name="ground_truth_2", activation='sigmoid')(conv9)
    
    model = Model(inputs=ImageInput, outputs=output)

    model.compile(loss=IOU_multi_loss, optimizer=Adam(lr = learning_rate))
    
    #load pretrained weights from keras 
    #https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    #weights_path = './model_weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    #model.load_weights(weights_path, by_name=True)
    
    # Load trained weights - if resuming training
    model_weights = "./model_weights/Best_FCN_Vgg19_unet_final.hdf5"
    model.load_weights(model_weights)
    
    print("Weights loaded")    

    return model

model = FCN_Vgg19_unet()
model_weights = "FCN_Vgg19_unet_"+test_name+".hdf5"    
    
model.summary()

# save model as json
json_string = model.to_json()
with open('./model_weights/'+use_model+"_"+test_name+'.json', 'w') as file:
    json.dump(json_string, file)

print("model saved as ", "./model_weights/"+use_model+'.json')

# create checkpoints for each improvement
# use IOU as val_loss
checkpoint = ModelCheckpoint(filepath = './model_weights/Best_'+model_weights, verbose = 1, save_best_only=True, monitor='val_loss')

# early termination with epochs as patience = 2 to prevent overfitting
callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# fits the model on batches with real-time data augmentation:
# https://keras.io/models/model/
history = model.fit_generator(train_data_gen,
                    steps_per_epoch = set_steps_per_epoch(data_size_train_road, batch_size), 
                    epochs = epochs, 
                    verbose = 1, 
                    callbacks = [checkpoint, callback], 
                    validation_data = val_data_gen,
                    validation_steps = set_steps_per_epoch(data_size_val_road, batch_size))

print("Training completed")

#Save model weights final
model.save_weights('./model_weights/Final_'+model_weights)
print("weights stored as ", 'Final_'+model_weights)

K.clear_session()
del model

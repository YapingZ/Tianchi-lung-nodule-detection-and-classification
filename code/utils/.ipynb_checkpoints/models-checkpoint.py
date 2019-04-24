# -*- coding: utf-8 -*-
import utils.helpers as helpers



import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import numpy as np
import pandas as pd
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
import os
import glob
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage
import sklearn
import time

from typing import List, Tuple
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from sklearn.utils import shuffle

from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Dropout, BatchNormalization,SpatialDropout2D,Convolution3D,MaxPooling3D, UpSampling3D, Flatten, Dense
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler,EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import plot_model

random.seed(1321)
numpy.random.seed(1321)

import warnings
warnings.filterwarnings("ignore")

K.set_image_dim_ordering('th') 

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return ((2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1))

def unet_model1(dropout_rate,width):
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="elu")(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(width*16, (3, 3), padding="same", activation="elu")(pool4)
    conv5 = BatchNormalization(axis = 1)(conv5)
    conv5 = Convolution2D(width*16, (3, 3), padding="same", activation="elu")(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = SpatialDropout2D(dropout_rate)(up6)
    conv6 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(conv6)
    conv6 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = SpatialDropout2D(dropout_rate)(up7)
    conv7 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(conv7)
    conv7 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = SpatialDropout2D(dropout_rate)(up8)
    conv8 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(conv8)
    conv8 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = SpatialDropout2D(dropout_rate)(up9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv9)
    conv10 = Convolution2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])
    
    #plot_model(model, to_file='model1.png',show_shapes=True)
    return model

def generate_train(start, end, seed = None):
    lung = sorted(glob.glob(src + 'lung/*.png'))[start:end]
    nod = sorted(glob.glob(src + 'nodule/*.png'))[start:end]    
    while True:
        print('Shuffling data')
        lung, nod = shuffle(lung, nod)
        for i in range(len(lung)):            
            lungs = cv2.imread(lung[i],cv2.IMREAD_GRAYSCALE).astype('float32')
            nods = cv2.imread(nod[i],cv2.IMREAD_GRAYSCALE).astype('float32')
            lungs = np.expand_dims(lungs,0)
            nods = np.expand_dims(nods,0)               
            lungs = np.expand_dims(lungs,0)
            nods = np.expand_dims(nods,0)            
            yield(lungs, nods)

            
def generate_val(start, end, seed = None):

    lung = sorted(glob.glob(src + 'lung/*.png'))[start:end]    
    nod = sorted(glob.glob(src + 'nodule/*.png'))[start:end] 
    while True:
        for i in range(len(lung)):            
            lungs = cv2.imread(lung[i],cv2.IMREAD_GRAYSCALE).astype('float32')
            nods = cv2.imread(nod[i],cv2.IMREAD_GRAYSCALE).astype('float32')  
            lungs = np.expand_dims(lungs,0)
            nods = np.expand_dims(nods,0)             
            lungs = np.expand_dims(lungs,0)
            nods = np.expand_dims(nods,0)
            yield(lungs, nods)



def unet_fit(name, start_t, end_t, start_v, end_v, check_name = None):
    
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 15, 
                                   verbose = 1),
    ModelCheckpoint('/Volumes/solo/ali/Data/model/{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    
    if check_name is not None:
        check_model = '/Volumes/solo/ali/Data/model/{}.h5'.format(check_name)
        model = load_model(check_model, 
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model()

    model.fit_generator(generate_train(start_t, end_t), nb_epoch = 30, verbose = 1, 
                        validation_data = generate_val(start_v, end_v),
                        callbacks = callbacks,
                        samples_per_epoch = 2600, nb_val_samples = 300)
        
    return

def unet_model():
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)    
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)    
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = BatchNormalization(axis = 1)(conv5)   
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
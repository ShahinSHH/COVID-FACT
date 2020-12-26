#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-FACT binary test code implementation.

!! Note: COVID-FACT framework is in the research stage. Use only for research ourposes at this time.
Don't use COVID-FACT as a replacement of the clinical test and radiologist review.

Created by: Shahin Heidarian, Msc. at Concordia University
E-mail: s_idari@encs.concordia.ca

** The code for the Capsule Network implementation is adopted from https://keras.io/examples/cifar10_cnn_capsule/.
"""

from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import cv2
from lungmask import mask #lung segmentation model
import SimpleITK as sitk

# Set the path based on your data directory
data_path = r'Path to the folder including dicom files/'

K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
        
       
           
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

# normalization function
def normalize_image(x): #normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))
        
def segment_lung(mask,model,volume_path):
    
    # model = mask.get_model('unet','R231CovidWeb')
    #loop through all dcm files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    
    dataset = pydicom.dcmread(lstFilesDCM[0]) # a sample image
    slice_numbers = len(lstFilesDCM) #number of slices
    # print('Slices:',slice_numbers)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        # print('Image size:',rows,cols)
    
    slice_z_locations = []
    for filenameDCM in lstFilesDCM:
        ds = pydicom.dcmread(filenameDCM)
        slice_z_locations.append(ds.get('SliceLocation'))
    
    #sorting slices based on z locations
    slice_locations = list(zip(lstFilesDCM,slice_z_locations))
    sorted_slice_locations = sorted(slice_locations, key = lambda x: x[1])[-1::-1]
    
    # Saving Slices in a numpy array
    ArrayDicom = np.zeros((slice_numbers,rows,cols))
    lung_mask = np.uint8(np.zeros((slice_numbers,rows,cols)))
    # loop through all the DICOM files
    i = 0
    for filenameDCM, z_location in sorted_slice_locations:
        # read the file
        ds = sitk.ReadImage(filenameDCM)
        segmentation = mask.apply(ds, model)
        lung_mask[i,:,:] = np.uint8(((segmentation>0)*1)[0])
        ArrayDicom[i, :, :] = sitk.GetArrayFromImage(ds)
        i = i+1
    
    lungs = np.zeros((ArrayDicom.shape[0],256,256,1))    
    # resizing the data
    for i in range(ArrayDicom.shape[0]):
        ct = normalize_image(ArrayDicom[i,:,:])
        mask_l = lung_mask[i,:,:]
        seg = mask_l * ct #apply mask on the image
        img = cv2.resize(seg,(256,256))
        img = normalize_image(img)
        lungs[i,:,:,:] = np.expand_dims(img,axis = -1)
    # print('Successfully segmented.')    
    return lung_mask, ArrayDicom, lungs

def test_one_dicom(model,X_test):
    # Test
    X_test_normal = np.zeros(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test_normal[i,:,:,:] = normalize_image(X_test[i,:,:,:])
   
    predict = model.predict([X_test_normal])
    predict = np.argmax(predict,axis=1)
    print('Abnormal slices:',sum(predict))    
    return X_test_normal, predict

def stage_one_output(model,X_test):
    X_test, predict_init = test_one_dicom(model,X_test) # predict classes for all slices      
    sum_seg = np.sum(np.sum(X_test,axis=1),axis=1) 
    a = np.where(sum_seg[:,0] != 0) # to find out if lung exists or not
    predict = predict_init[a]
    infected_slices = sum(predict)#save the number of infected slices
    case_slices = len(predict)
    print('Total Slices:',case_slices)
    
    X_numpy_inf = np.zeros((infected_slices,256,256,1))
    X_numpy_hlt = np.zeros((case_slices-infected_slices,256,256,1))
    hlt_ind = 0
    inf_ind = 0
    for i in range(X_test.shape[0]):
        if sum_seg[i,0] != 0 : #if lung exists
            if predict_init[i] == 0: #save healthy slices
                X_numpy_hlt[hlt_ind,:,:,:] = X_test[i,:,:,:]
                hlt_ind = hlt_ind+1
            if predict_init[i] == 1: #save healthy slices
                X_numpy_inf[inf_ind,:,:,:] = X_test[i,:,:,:]
                inf_ind = inf_ind+1 
    
    return X_numpy_hlt,X_numpy_inf,len(predict_init),infected_slices,case_slices

def stage_two_output(x_test,normal_thresh,model2,cutoff):
    
    if (len(x_test)<=0.03*normal_thresh):
        pred_final = 0
        prob_one = 0
        # print('A nonCOVID case.')
    else:    
        pred = model2.predict(x_test)
        prob_one = sum(pred[:,1])/len(pred[:,1])
        pred_final = (prob_one>=cutoff)*1 # cut-off probability  
    return prob_one, pred_final

#%% Model Stage 1

input_image = Input(shape=(None, None, 1))
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
# # # # # # # # # # # # # # #
x = Reshape((-1, 128))(x)
x = Capsule(32, 16, 1, True)(x)  
x = Capsule(32, 16, 1, True)(x)   
capsule = Capsule(2, 16, 1, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)


model1 = Model(inputs=[input_image], outputs=[output])

# adam = optimizers.Adam(lr=1e-4) 
# model1.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model1.summary()
model1.load_weights('weights-stage1-final-91.h5')

#%% Model Stage 2

input_image = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation='relu',trainable = True)(input_image)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(32, (3, 3), activation='relu',trainable = True)(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu',trainable = True)(x)
x = Conv2D(64, (3, 3), activation='relu',trainable = True)(x)
x = Dropout(0.2) (x)
# # # # # # # # # # # # # # #
x = Reshape((-1, 64))(x) 
x = Capsule(16, 8, 1, True)(x)   
capsule = Capsule(2, 16, 1, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)


model2 = Model(inputs=[input_image], outputs=[output])

# adam = optimizers.Adam(lr=1e-4) 
# model2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model2.summary()
model2.load_weights('weights-stage2-final-99.h5') 

#%%
# lstFolders = sorted(os.listdir(data_path))
model = mask.get_model('unet','R231CovidWeb')

#%% Testing
print('Segmenting lung area...')
lung_mask, ArrayDicom, lung = segment_lung(mask,model,data_path)
print('Segmentation is Completed.')
    
# Stage 1
X_numpy_hlt, X_numpy_inf, slice_size, infected_slices,case_slices = stage_one_output(model1,lung)

# Stage 2  
normal_thresh = slice_size
x_test = X_numpy_inf
cutoff = 0.5 #cut-off probability
prob_one, pred_final =  stage_two_output(x_test,normal_thresh,model2,cutoff) 
if pred_final==1 :
    prediction = 'COVID-19'
else:
    prediction = 'non-COVID'
    
print('prediction: ',prediction)
      

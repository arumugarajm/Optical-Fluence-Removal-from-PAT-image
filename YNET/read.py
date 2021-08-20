# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:01:08 2021

@author: arumugaraj
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io



def read_data(path):
    i1=3   # Number of training dataset
    i2=2  # Number of validation dataset Ex.(16-12=4) 4 dataset used for validation, remaining datset used for testing

    GT_images = []  # Ground truth images are stored
    input_images = []   # input images are stored
    raw = []
    

    for file_name in glob.glob(path):   # To read the dataset
        # print(file_name) # To display the dataset name
        mat=scipy.io.loadmat(file_name) # Read the mat file
        a=mat['Xi']  # Extract Xi variable from mat file
        b=mat['Xim'] # Extract GT variable from mat file
        c=mat['sdn2'] # Extract raw data from mat file
        A = (a-np.amin(a))/(np.max(a)-np.min(a))#Normalized values between o to 1
        B = (b-np.amin(b))/(np.max(b)-np.min(b))
        C = (c-np.amin(c))/(np.max(c)-np.min(c))
        
        X = np.asarray(B)   # Input and GT stored as array
        Y =np.asarray(A)
        Z = np.asarray(C)
        GT_images.append(X)
        input_images.append(Y)
        raw.append(Z)
        # minvalue = numpy.amin(arr2D)
        # maxvalue = numpy.amax(arr2D)
        # range = maxvalue-minvalue
        # new = (arr2D-minvalue)/range
        # nor=(arr2D-minvalue)/(maxvalue-minvalue)
        # mean = numpy.average(arr2D)
        # dev = numpy.std(arr2D)
        # neww = (arr2D - mean) / (dev)
       
    
    GT_images = np.asarray(GT_images)
    input_images = np.asarray(input_images)
    raw = np.array(raw)
    X1 = np.expand_dims(input_images[: i1], axis=-1).astype('float32') #train data
    X2 = np.expand_dims(input_images[i1:i2], axis=-1).astype('float32') #validation data
    X3 = np.expand_dims(input_images[i2:], axis=-1).astype('float32') #test data
    
    Y1 = np.expand_dims(raw[: i1], axis=-1).astype('float32')
    Y2 = np.expand_dims(raw[i1: i2], axis=-1).astype('float32')
    Y3 = np.expand_dims(raw[i2: ], axis=-1).astype('float32')
    
    Z1 = np.expand_dims(GT_images[: i1], axis=-1).astype('float32')
    Z2 = np.expand_dims(GT_images[i1: i2], axis=-1).astype('float32')
    Z3 = np.expand_dims(GT_images[i2:], axis=-1).astype('float32')

    return X1,X2,X3,Y1,Y2,Y3,Z1,Z2,Z3

def print_data_samples(input_X, input_Y, input_Z):
    """
    Prints 4 pair of images (Y, X), where:
    input_Y --- ground truth
    input_X --- noisy approximation of Y
    """
    plt.figure(figsize=(12., 6.))

    for i in range(1,5):
        #number of different inputs to display give as row; number of inputs from a single type give as col
        plt.subplot(3, 4, i)
        plt.imshow(input_X[i,:,:,0], cmap='gray') # input1, input 2 and GT
        plt.axis('off')
        plt.subplot(3, 4 , i + 4)
        plt.imshow(input_Y[i,:,:,0], cmap='gray')
        plt.axis('off')
        plt.subplot(3,4, i+8)
        plt.imshow(input_Z[i,:,:,0], cmap='gray')
        plt.axis('off')

    plt.show()


def print_model_outputs(model, input_X,input_Y, input_Z, title):
    """
    Prints 4 triple of images (prediction, input_X, input_Y), where:
    input_Y --- ground truth
    input_X --- noisy approximation of Y
    prediction = model(input_X) --- cleaned X
    """
    prediction = model(input_X,input_Y,input_Z, training=False)

    fig = plt.figure(figsize=(9., 9.))

    for i in range(1,4):
        plt.subplot(3,3, (i - 1) * 3+ 1)
        plt.imshow(prediction[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=42, y=-4, s='Ynet')
        plt.subplot(3,3, (i - 1) * 3+ 2)
        plt.imshow(input_X[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=56, y=-4, s='Input')
        plt.subplot(3,3, (i - 1) * 3 + 3)
        plt.imshow(input_Z[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=36, y=-4, s='Ground Truth')
    
    fig.suptitle(title, fontsize=16, y=0.94)
    plt.show()

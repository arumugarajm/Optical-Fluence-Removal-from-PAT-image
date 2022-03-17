# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:01:08 2021

@author: arumugaraj
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io


###### To read all .mat file for training and validation ##########
def read_data(path):
    i1= 7400 # Number of training dataset
    GT_images = []  # Ground truth images are stored
    input_images = []   # input images are stored
    

    for file_name in glob.glob(path):   # To read the dataset
        # print(file_name) # To display the dataset name
        mat=scipy.io.loadmat(file_name) # Read the mat file
        ##### Extract necessary variables from the mat file
        a=mat['Xim'] # Extract GT variable from mat file
        b=mat['Xi']  # Extract Xi variable from mat file

        A = (a-np.min(a))/(np.max(a)-np.min(a))#Normalized values between o to 1
        B = (b-np.min(b))/(np.max(b)-np.min(b))
        X = np.asarray(A)   # Input and GT stored as array
        Z =np.asarray(B)
        A1 = np.asarray(a)
        B1 = np.asarray(b)
    
        GT_images.append(X)
        input_images.append(Z)

    GT_images = np.asarray(GT_images)
    input_images = np.asarray(input_images)
   
    X1 = np.expand_dims(input_images[: i1], axis=-1).astype('float32') #train data
    X2 = np.expand_dims(input_images[i1:], axis=-1).astype('float32') #validation data
 
    Z1 = np.expand_dims(GT_images[: i1], axis=-1).astype('float32') #train data
    Z2 = np.expand_dims(GT_images[i1:], axis=-1).astype('float32') # validattion data

    return X1,X2,Z1,Z2

################## To visualize few input data ################

def print_data_samples(input_X, input_Y):
    """
    Prints 4 pair of images (Y, X), where:
    input_Y --- ground truth
    input_X --- noisy approximation of Y
    """
    plt.figure(figsize=(12., 6.))

    for i in range(1,5):
        plt.subplot(2, 4, i)
        plt.imshow(input_X[i,:,:,0], cmap='gray')
        plt.axis('off')
        plt.subplot(2, 4, i + 4)
        plt.imshow(input_Y[i,:,:,0], cmap='gray')
        plt.axis('off')

    plt.show()

################# To visulalize few outputs generated from the model #################
def print_model_outputs(model, input_X, input_Y, title):
    """
    Prints 4 triple of images (prediction, input_X, input_Y), where:
    input_Y --- ground truth
    input_X --- noisy approximation of Y
    prediction = model(input_X) --- cleaned X
    """
    prediction = model(input_X, training=False)

    fig = plt.figure(figsize=(9., 9.))

    for i in range(1,4):
        plt.subplot(3, 3, (i - 1) * 3 + 1)
        plt.imshow(prediction[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=42, y=-4, s='DeepResUnet')
        plt.subplot(3, 3, (i - 1) * 3 + 2)
        plt.imshow(input_X[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=56, y=-4, s='Input')
        plt.subplot(3, 3, (i - 1) * 3 + 3)
        plt.imshow(input_Y[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=36, y=-4, s='Ground Truth')
    
    fig.suptitle(title, fontsize=16, y=0.94)
    # font1 = {'family':'serif','color':'k','size':18}
    plt.imsave('prediction.png',prediction[1,:,:,0],cmap = 'gray')
    # plt.title('U-Net',fontdict = font1,fontweight="bold")
    
    plt.imsave('inputX.png', input_X[1,:,:,0],cmap='gray')
    # plt.title('TR',fontdict = font1,fontweight="bold")
    plt.imsave('GT.png',input_Y[1,:,:,0], cmap='gray')
    # plt.title('Grount truth',fontdict = font1,fontweight="bold")
    plt.show()
    
    
    
    
    
########### To save single predicted result ###############

    fig = plt.figure(figsize=(12., 6.))
    
    error = input_Y[1,:,:,0] - input_X[1,:,:,0]
    plt.imsave('errorinput.png', error,cmap='gray')
    plt.imshow(error, cmap='gray')
    plt.axis('off')
    
    error1 = input_Y[1,:,:,0]-prediction[1,:,:,0]
    plt.imsave('error1output.png', error,cmap='gray')
    plt.imshow(error1,cmap='gray')
    plt.axis('off')
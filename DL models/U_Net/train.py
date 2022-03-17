# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:25:52 2021

@author: arumugaraj
"""

import tensorflow as tf
from Unet import Unet,avg_NSR,avg_log_SNR
from read import read_data, print_data_samples, print_model_outputs
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger
import matplotlib.pyplot as plt
import glob
import scipy.io
import numpy as np
from skimage.measure import profile_line
from PIL import Image
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator

############# To read testing datasets ###############
path1 = "/media/fistlab/raj/all/35/*.*"

Z3 = [] # GT for testing data 
X3 = [] # testing input image domain

for file_name in glob.glob(path1):   # To read the dataset
#   print(file_name) # To display the dataset name
    mat=scipy.io.loadmat(file_name) # Read the mat file
    ##### Extract necessary variables from the mat file
    R1 = mat['Xi']
    R11=mat['Xim']
    X = np.asarray(R1[:,:]) 
    B1 = np.asarray(R11[:,:])
    B1 = (B1-np.amin(B1))/(np.amax(B1)-np.amin(B1))
    A = (X-np.amin(X))/(np.amax(X)-np.amin(X))
    X3.append(A)
    Z3.append(B1)

   
X3 = np.asarray(X3)
Z3 = np.asarray(Z3)
R1 = np.expand_dims(X3, axis=-1).astype('float32') #train data
Z3 =np.expand_dims(Z3, axis=-1).astype('float32')



############# Give path for the training data #################
################# read the data ########################
path = "/media/fistlab/raj/combined dataset/*.*"
X1,X2,Z1,Z2= read_data(path)

############### print few samples from the training set #################
print_data_samples(X1, Z1)

################### create an instance of the U-Net model ################
model = Unet(input_size=256, input_channels=1, filters=64)

############ set up checkpoint ####################
early_stop = EarlyStopping(monitor='val_loss', patience=12, verbose=1) #early stop incase no improvement in the validation loss
checkpoint = ModelCheckpoint(filepath='./model_checkpoints/', save_best_only=True, save_weights_only=True,monitor='val_avg_log_SNR', mode='max', verbose=1) # store the weights
log_csv = CSVLogger('my_logs.csv', separator=',', append=False) #### To store the train and validation loss
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                               patience=6, min_lr=0.0000001) # reduce the learning rate incase we want
callbacks_list = [checkpoint, log_csv,reduce_lr] ### If we need early stop add that parameter here 

################ train the model #####################
start = datetime.now()

history = model.fit(X1, Z1, validation_data=(X2, Z2), batch_size=6, epochs=150, callbacks=callbacks_list)

end = datetime.now()
time_taken = end - start
print('Time: ',time_taken) 

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

############## To test the model ######################

############## load the weights of the model with best performance on the validation set ###################
model.load_weights('./model_checkpoints/')

start1= datetime.now()

# check the performance of unet on the testing set
y_pred = model.predict(X3)
end1 = datetime.now()
time_taken1 = end1 - start1
print('Time1: ',time_taken1) 

############ check the performance of Output on the testing set ###############
print('average PSNRLoss of U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(Z3, y_pred, 1.)).numpy()))
print('variance PSNR of U-Net on the testing set: {}'.format(tf.math.reduce_variance(tf.image.psnr(Z3, y_pred, 1.)).numpy()))
print('average SSIM of U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(Z3, y_pred, 1.)).numpy()))
print('variance SSIM of U-Net on the testing set: {}'.format(tf.math.reduce_variance(tf.image.ssim(Z3, y_pred, 1.)).numpy()))

############ check the performance of Input on the testing set ###############
print('average PSNRLoss of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(Z3, X3, 1.)).numpy()))
print('variance PSNR of Input on the testing set: {}'.format(tf.math.reduce_variance(tf.image.psnr(Z3, X3, 1.)).numpy()))
print('average SSIM of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(Z3, X3, 1.)).numpy()))
print('variance SSIM of Input on the testing set: {}'.format(tf.math.reduce_variance(tf.image.ssim(Z3, X3, 1.)).numpy()))



# print few results of outputs of the U-Net on the training set
print_model_outputs(model, X1[:4], Z1[:4], 'Training Data')

# print few results of outputs of the U-Net on the validation set
print_model_outputs(model, X2[:4], Z2[:4], 'Validation Data')

# # print few results of outputs of the model on the testing set
print_model_outputs(model, X3[:4], Z3[:4], 'Testing Data')







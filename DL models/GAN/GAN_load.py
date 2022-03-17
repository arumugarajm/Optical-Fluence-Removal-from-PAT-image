#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:40:45 2021

@author: fistlab
"""

import keras
import tensorflow as tf

def avg_NSR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(noise / signal)

def avg_log_SNR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(20. * tf.math.log(signal / noise) / tf.math.log(10.0))


model = keras.models.load_model('/media/fistlab/raj/GAN/own/resgan/stored models/A/agen_model165.h5', custom_objects = {"avg_NSR": avg_NSR})
# model.summary()
# model = keras.models.load_model('/media/fistlab/raj/GAN/own/resgan/3Dmodel/Agen_model105.h5', custom_objects = {"avg_NSR": avg_NSR})
# from model import Generator, Discriminator
from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
# from read import read_data,print_model_outputs
# import read
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import keras.backend as K
from tensorflow.keras.optimizers import Adam
# from read import read_data, print_data_samples, print_model_outputs#,plot_generated_images
from numpy.random import randint
from numpy import zeros
from numpy import ones
from numpy import vstack
from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
import glob
from skimage.measure import profile_line


# ######### 2D model testing  #########################

path0 = "/media/fistlab/Backup Plus/TEST/800/*.*"

### Store GT and Input images with normalization
GT0_im = []
input_images0 = []
########
### Store GT and Input images without normalization
GT0w_im = []

R0w = []



for file_name in glob.glob(path0):   # To read the dataset
# print(file_name) # To display the dataset name
    mat=scipy.io.loadmat(file_name) # Read the mat file
    R0 = mat['Xi']
    R00=mat['Xim']
    A0 = np.asarray(R0) 
    B0 = np.asarray(R00)
    BB0 = (B0-np.min(B0))/(np.max(B0)-np.min(B0))
    AA0 = (A0-np.min(A0))/(np.max(A0)-np.min(A0))
    input_images0.append(AA0)
    GT0_im.append(BB0)
    R0w.append(R0)
    GT0w_im.append(R00)
    
R0w = np.asarray(R0w)
GT0w_im = np.asarray(GT0w_im)
input_images0 = np.asarray(input_images0)
GT0_im = np.asarray(GT0_im)

R0w = np.expand_dims(R0w, axis=-1).astype('float32')
GT0w_im = np.expand_dims(GT0w_im, axis=-1).astype('float32')
R0 = np.expand_dims(input_images0, axis=-1).astype('float32') #train data
GT0_im =np.expand_dims(GT0_im, axis=-1).astype('float32')
# ###############################################

test_GT = GT0_im
test_input = R0
generated_images_sr = model.predict(test_input)
s = 8   # 0 for SNR 8 for wavelength
i0 = R0w[s,4:,4:]
o0 = GT0w_im[s,4:,4:]
i0m = np.min(i0)
i0ma = np.max(i0)
o0m = np.min(o0)
o0ma = np.max(o0)


y_pre0 = generated_images_sr[s,4:,4:]
y_pre0 = y_pre0*(o0ma-o0m)+o0m
print('average PSNRLoss of Input on the testing set1: {}'.format(tf.math.reduce_mean(tf.image.psnr(o0, i0, 1.)).numpy()))
print('average SSIM of Input on the testing set1: {}'.format(tf.math.reduce_mean(tf.image.ssim(o0, i0, 1.)).numpy()))

print('average PSNRLoss of Input on the testing set1: {}'.format(tf.math.reduce_mean(tf.image.psnr(o0, y_pre0, 1.)).numpy()))
print('average SSIM of Input on the testing set1: {}'.format(tf.math.reduce_mean(tf.image.ssim(o0, y_pre0, 1.)).numpy()))

print ('-'*15, 'Space', '-'*15)
# # ###############################



    
################ print_model_outputs1 ##########################
plt.imsave('R0input.png',R0[s,4:,4:,0],cmap = 'gray')
plt.imsave('R0prediction.png', generated_images_sr[s,4:,4:,0],cmap='gray')
# plt.imsave('ER1.png',y_predict1[1,:,:,0]-R1[1,:,:,0],cmap='gray')
plt.imsave('ERPR0.png',GT0_im[s,4:,4:,0]-generated_images_sr[s,4:,4:,0],cmap='gray')
plt.imsave('R0GT.png',GT0_im[s,4:,4:,0],cmap='gray')


# SNR_30 = y_pre0 
# SNR_35 = y_pre0 
# SNR_40 = y_pre0 
# nm_600 = y_pre0 
# nm_700 = y_pre0 
# nm_800 = y_pre0 

# scipy.io.savemat('/media/fistlab/raj/2Dlineprofile/GANnew.mat', dict(SNR_30 = SNR_30, SNR_35 = SNR_35, SNR_40 = SNR_40, nm_600 =nm_600, nm_700 =nm_700, nm_800 =nm_800))







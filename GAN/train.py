# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:25:52 2021

@author: arumugaraj
"""

import tensorflow as tf
from model import Generator,Discriminator
from read import read_data, print_data_samples, print_model_outputs
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# read the data
path = "D:\\project\\Dataset\\New folder\\*.*"
X1,X2,X3,Z1,Z2,Z3 = read_data(path)

# print few samples from the training set
print_data_samples(X1, Z1)














# # load the weights of the model with best performance on the validation set
# model.load_weights('./model_checkpoints/')

# # check the performance of unet on the testing set
# y_pred = model.predict(X3)
# # print('average PSNRLoss of U-Net on the testing set: {}'.format(avg_log_SNR(test_set_Y, y_pred).numpy()))
# print('average PSNRLoss of U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(Z3, y_pred, 1.)).numpy()))
# print('average SSIM of U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(Z3, y_pred, 1.)).numpy()))

# # check the performance of Input on the testing set
# # print('average PSNRLoss of Input on the testing set: {}'.format(avg_log_SNR(test_set_Y, test_set_X).numpy()))
# print('average PSNRLoss of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(Z3, X3, 1.)).numpy()))
# print('average SSIM of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(Z3, X3, 1.)).numpy()))

# # print few results of outputs of the U-Net on the training set
# print_model_outputs(model, X1[:4], Z1[:4], 'Training Data')

# # print few results of outputs of the U-Net on the validation set
# print_model_outputs(model, X2[:4], Z2[:4], 'Validation Data')

# # print few results of outputs of the model on the testing set
# print_model_outputs(model, X3[:4], Z3[:4], 'Testing Data')



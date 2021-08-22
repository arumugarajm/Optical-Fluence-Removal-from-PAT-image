# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:21:01 2021

@author: arumugaraj
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense, BatchNormalization,Activation,LeakyReLU, Conv2DTranspose, Input, concatenate,  MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt

def Act(x):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # x = Activation("relu")(x)
    return x
def conv(x,filters):
    
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Act(x)
    p = MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    return x,p

def conv1(x,skip,filters):
    x =  Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(x)
    x = concatenate([x,skip])
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    
    return x

input_size = (256,256,1)   
def Generator(input_sizeg):
    filters = 64
    #encoder
    inputs = Input(shape=(input_sizeg))
    print(inputs.shape)
    x1, e1 = conv(inputs,filters)
    print(x1.shape,e1.shape)
    x2, e2 = conv(e1,2*filters)
    print(x2.shape,e2.shape)
    x3, e3 = conv(e2,4*filters)
    print(x3.shape,e3.shape)
    x4, e4 = conv(e3,8*filters)
    print(x4.shape,e4.shape)
    x5, e5 = conv(e4,16*filters)
    print(x5.shape,e5.shape)
    
    #bridge
    bridge = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same')(e5)
    x=Act(bridge)

    #decoder
    d1 = conv1(x,x5,16*filters)
    d2 = conv1(d1,x4,8*filters)
    d3 = conv1(d2,x3,4*filters)
    d4 = conv1(d3,x2,2*filters)
    d5 = conv1(d4,x1,filters)
    
    #output
    output = Conv2D(1, (1, 1), padding="same", activation="relu")(d5)
    
    
    
    model = Model(inputs= inputs, outputs=output)
    # model.compile(optimizer=Adam(lr = 1e-4), loss = 'mse', metrics = ['mse'])
    return model  


def conv2(x,filters):
    x = Conv2D(filters=filters, kernel_size=4, strides=2, padding='same')(x)
    x = Act(x)
    return x
    

#### Discriminator
def Discriminator(input_sized):
    filters = 64
    
    inputs = Input(shape=(input_size))
    
    x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(inputs)
    x = LeakyReLU(alpha = 0.2)(x)
    # print(inputs.shape)
    x = conv2(inputs,filters)
    # print(x.shape)
    x = conv2(x,2*filters)
    x = conv2(x,4*filters)
    x = conv2(x,8*filters)
    
    
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha = 0.2)(x)
       
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model





#compile discriminator
    
input_sizeg = (256,256,1) #train X1


# create and compile discriminator model
Dis = Discriminator(input_size)
Dis.compile ( loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
# Dis.summary()

# create generator model
Gen = Generator(input_size)
Gen.summary()

# pass input to generator
img = Generator(input_sizeg)

Discriminator.trainable = False

fake_pred = Discriminator(img)

# Create the combined model object
combined_model_gen = Model(input_sizeg, fake_pred)

# Compile the combined model
combined_model_gen.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Config

batch_size = 32
epochs = 60
sample_period = 200 # every `sample_period` steps generate and save some data"


# Create batch labels to use when calling train_on_batch
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

# Store the losses
d_losses = []
g_losses = []

# Create a folder to store generated images
if not os.path.exists('gan_images'):
  os.makedirs('gan_images')

H=W=256

# A function to generate a grid of random samples from the generator and save them to a file

def sample_images(epoch):
  rows, cols = 5, 5
  noise = input_sizeg
  imgs = Generator.predict(noise)

  # Rescale images 0 - 1
  # imgs = 0.5 * imgs + 0.5

  fig, axs = plt.subplots(rows, cols)
  idx = 0
  for i in range(rows):
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig("gan_images/%d.png" % epoch)
  plt.close()   
  
  
  
# Main training loop
for epoch in range(epochs):
  ###########################
  ### Train discriminator ###
  ###########################
  
  # Select a random batch of images
  idx = np.random.randint(0, x_train.shape[0], batch_size)
  real_imgs = x_train[idx]
  
  # Generate fake images
  noise = np.random.randn(batch_size, latent_dim)
  fake_imgs = Generator.predict(noise)
  
  # Train the discriminator
  # both loss and accuracy are returned
  d_loss_real, d_acc_real = Discriminator.train_on_batch(real_imgs, ones)
  d_loss_fake, d_acc_fake = Discriminator.train_on_batch(fake_imgs, zeros)
  d_loss = 0.5 * (d_loss_real + d_loss_fake)
  d_acc  = 0.5 * (d_acc_real + d_acc_fake)
  
  
  #######################
  ### Train generator ###
  #######################
  
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model_gen.train_on_batch(noise, ones)
  
  # do it again!
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model_gen.train_on_batch(noise, ones)
  
  # Save the losses
  d_losses.append(d_loss)
  g_losses.append(g_loss)
  
  if epoch % 10 == 0:
    print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, \
      d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
  
  if epoch % sample_period == 0:
    sample_images(epoch)
    

plt.plot(g_losses, label='g_losses')
plt.plot(d_losses, label='d_losses')
plt.legend()


from skimage.io import imread
i = imread('gan_images/0.png')
plt.imshow(i)
    
    
# if __name__ == "__main__":
    # model = Generator(input_size)
    # model1 = Discriminator((256,256,1))
    # model.summary()
    # model1.summary()
   
    
       
from model import Generator, Discriminator
from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
from read import read_data,print_model_outputs
import read
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from read import read_data, print_data_samples, print_model_outputs#,plot_generated_images
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

########## Loss fuction to train the model ##################
def avg_NSR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(noise / signal)


############# To read training and testing datasets ###############
image_shape = (256,256,1)
path = "/media/fistlab/DATA/breast dataset/kwavenew/train1/*.*"
X1,Z1 = read_data(path) 
GT = Z1
fluence_affected = X1


 

############# To store the GAN model ###########################
model_save_dir = '/media/fistlab/raj/GAN/own/resgan/3Dmodel/A'
output_dir = '/media/fistlab/raj/GAN/own/resgan/3Dmodel/A'

################ Create GAN model #####################
def GAN(gen,dis,shape):
    dis.trainable = False
    optimizer = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    gan_input = Input(shape=shape)
    x = gen(gan_input)
    gan_output = dis(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[avg_NSR, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    # gan.summary()
    return gan


################ train the model #####################

def train(epochs, batch_size):
  
    
    batch_count = len(Z1)//2
    shape = (256,256,1)
    image_shape =(256,256,1)
    
    generator = Generator(shape)
    discriminator = Discriminator(image_shape)

    optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=avg_NSR, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    gan = GAN(generator,discriminator, shape)
    gan.summary()
    
    loss_file = open(model_save_dir + 'losses.txt' , 'w+')
    loss_file.close()
  

    for e in range(1, epochs+1):
            print ('-'*15, 'Epoch %d' % e, '-'*15)
            for i in tqdm(range(batch_count)):
                
                rand_nums = np.random.randint(0, len(GT), size=batch_size)
                
                image_batch_GT = GT[rand_nums]
                image_batch_fluence = fluence_affected[rand_nums]

                preicted_image = generator.predict(image_batch_fluence)

                real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
                fake_data_Y = np.random.random_sample(batch_size)*0.2
                
                discriminator.trainable = True
                
                d_loss_real = discriminator.train_on_batch(image_batch_GT, real_data_Y)
                d_loss_fake = discriminator.train_on_batch(preicted_image, fake_data_Y)
                discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                

                image_batch_GT = GT[rand_nums]
                image_batch_fluence = fluence_affected[rand_nums]

    
                gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
                discriminator.trainable = False
                gan_loss = gan.train_on_batch(image_batch_fluence, [image_batch_GT,gan_Y])
                
            
           
            print("discriminator_loss : %f" % discriminator_loss)
            print("gan_loss :", gan_loss)
            gan_loss = str(gan_loss)
            
            loss_file = open(model_save_dir + 'losses.txt' , 'a')
            loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
            loss_file.close()
            
                
            if e % 5 == 0:
                generator.save(model_save_dir + 'gen_model%d.h5' % e)
                # discriminator.save(model_save_dir + 'dis_model%d.h5' % e)



                
                

    return  preicted_image,image_batch_GT,image_batch_fluence,real_data_Y,fake_data_Y,gan_Y,generated_image,rand_nums

preicted_image,image_batch_GT,image_batch_fluence,real_data_Y,fake_data_Y,gan_Y,generated_image,rand_nums  = train(500,3)
      
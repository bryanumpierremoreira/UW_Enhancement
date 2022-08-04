#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 21:22:43 2021

@author: Claudio D. Mello Jr.
Federal University of Rio Grande
Rio Grande - RS, Brazil

Article: Underwater Enhancement based on a self-learning strategy and attention mechanism 
         for high-intensity regions
Authors: Claudio D. Mello Jr., Bryan U. Moreira, Paulo J. O. Evald, Paulo L. Drews Jr. and Sivia S. Botelho         

Software verions:
    Python -> 3.7.6
    Tensorflow -> 2.20
    Keras -> 2.3.1
    Python IDE -> Spyder vs. 4.0.1  (Anaconda 1.10.0)

The training was performed in a computer with Core i5-6400 CPU, 32 MB RAM, Titan X GPU.    
O.S. Ubuntu 18.04.5 LTS
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model  #, load_model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from datetime import datetime
import pickle
import numpy as np
import os

#agora = datetime.now()
#hora_inicio = agora.strftime("%H:%M:%S")

# Load dataset


"""The dataset is loaded as a single file, containing images of 256x256x3 and scaled to [0., 1.0] for
training and test.
"""
os.chdir(dir_data)               # Directory with the dataset file
f = open('filename', 'rb')       # name of the dataset file
data = pickle.load(f)
f.close()
data = np.float32(data)

os.chdir(diretorio)
# dimensions of our images.
ni, ht, wd, ch = np.shape(data)
input_shape = (ht, wd, ch)

#Split dataset
(data_train), (data_test) = train_test_split(data, test_size=0.1)

#=============================================================================
# 

fe1 = [48, 48, 36, 36]   
fd = [36, 36, 48, 48]    
alpha = 0.19
alpha0 = 0.025
alpha1 = 0.09
eps = 1.0e-7
seedy1 = 17    
seedy2 = 31
seedy3 = 43    
uu = 0.02e-6
kkr = 15e-6  
bbr = 1.5e-6
regk = regularizers.l1(kkr)
regb = regularizers.l1(bbr)
batch_size = 6  


def escala(img):
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    a = tf.constant([-1.0], dtype=float)
    vm = tf.math.multiply(a, vmin)
    #vm = K.clip(vm, 0., 0.3)
    imgx = vm + img
    nmax = tf.reduce_max(imgx, axis=(1,2), keepdims=True)
    nmax = K.clip(nmax, 1.0, vmax)
    imgx = tf.math.divide_no_nan(imgx, nmax)
    return imgx

# ==============================  AUTOENCODER ================================

#Encoder ----------------------------------------------------------
#Block 256 --------------------------------------------------------

input_img = Input(shape=input_shape, dtype=tf.float32, name='Entrada')

x1 = Conv2D(fe1[0],
    (3,3),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(input_img) 

x1 = Conv2D(fe1[0],
    (3,3),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1) 

x1 = Conv2D(fe1[0],
    (3,3),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1) 

x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1) 
x256e1 = x1


#Fim do Bloco 256 --------------------------------------------------------

#Bloco 128 --------------------------------------------------------
x1 = Conv2D(fe1[0],
    (2,2),
    strides=(2, 2),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)    #128


x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)    #128

x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)           #128


x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)           #128
x128e1 = x1

#FIM bloco 128-e1 ---------------------------------------------

# Bloco 64-e1 -------------------------------------------------
x1 = Conv2D(fe1[1],
    (2,2),
    strides=(2, 2),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)


x1 = Conv2D(fe1[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)      #64x64
x64e1 = x1

#FIM bloco 64-e1 ----------------------------------------------------


#Bloco 32-e1 --------------------------------------------------------
x1 = Conv2D(fe1[2],
    (2,2),
    strides=(2, 2),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)       #32

x1 = Conv2D(fe1[2],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[2],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[2],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)
x32e1 = x1

#FIM bloco 32-e1 --------------------------------------------------

#Bloco 16-e1 ------------------------------------------------------
x1 = Conv2D(fe1[3],
    (2,2),
    strides=(2, 2),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(x1)       #16
x16e1 = x1

x1 = Conv2D(fe1[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(x1)       #16

x16e2 = x1

x1 = Conv2D(fe1[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(x1)       #16


xd = x1

##############################################################################
#Decoder  -----------------------------------------------------
#Bloco 16-d ---------------------------------------------------
xd = Conv2D(fd[0],
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)


xd = Concatenate()([xd, x16e2])
xd = Conv2D(fd[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)


xd = Concatenate()([xd, x16e1])
xd = Conv2D(fd[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

# Fim do bloco 16-d -----------------------------------------------------------

#Bloco 32-d ---------------------------------------------------

xd = UpSampling2D((2, 2))(xd)       #16 -> 32


xd = Concatenate()([xd, x32e1])      #<<<<<<<<<<Concatenacao

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+5),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+5),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)


#FIM bloco 32-d -------------------------------------------------

xd = UpSampling2D((2, 2))(xd)       #32 -> 64

#Bloco 64-d ----------------------------------------------------

xd = Concatenate()([xd, x64e1])

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+7),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+7),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+8),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+8),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

# xd = BatchNormalization(axis=-1, epsilon = eps)(xd)

#FIM bloco 64-d -------------------------------------------------

#Bloco 128-d ----------------------------------------------------
xd = UpSampling2D((2, 2))(xd)        #64 -> 128

#******* Switch conc 1 *******
# if random.randint(1,2) == 1:
#     xc128 = x128e1
# else: xc128 = x128e1
xd = Concatenate()([xd, x128e1])      #<<<<<<<<<<

xd = Conv2D(fd[3],
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
#xd = LeakyReLU(alpha)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
#xd = LeakyReLU(alpha0)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

#FIM bloco 128-d -------------------------------------------------------------

#Bloco 256 - d  --------------------------------------------------------------
xd = UpSampling2D((2, 2))(xd)        #64 -> 128

#xcc = xd
#skip 256
xd = Concatenate()([xd, x256e1])      #<<<<<<<<<<

xd = Conv2D(fd[3],
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+10),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+10),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)


xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+12),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+12),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
#fim do bloco 256 - d --------------------------------------------------------

# xd = BatchNormalization(axis=-1, epsilon = eps)(xd)
xd = Conv2D(3,
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
xd = LeakyReLU(alpha)(xd)

output_img0 = Conv2D(3,
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    name="SD0"
)(xd)
output_img = LeakyReLU(alpha, name='SD1')(output_img0)

output_img = Lambda(escala, name = 'SD2')(output_img)

#================================= OTIMIZADOR ================================
opt = optimizers.Adam(lr = 0.00007, 
                      beta_1=0.9, 
                      beta_2=0.999, 
                      epsilon = 1.1e-7, 
                      amsgrad=False)
                      
# ============================================================================
# =============================== Median =====================================
def get_median(ix):
    median = tfp.stats.percentile(ix, 50.0, axis=(1,2),
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    # median = K.reshape(median, (nbatch, 1, 1, 3))
    return median

# ============================================================================

# ============================================================================    

def vals(img):
    nbatch = K.shape(img)[0]
    vmdn = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    vmed = tf.reduce_mean(img, axis=(1,2), keepdims=True)
    bw = K.clip((vmax - vmin), 0.001, 1.0)
    return vmax, vmin, bw, vmed, vmdn


def haze(img):
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    bw = K.clip((vmax - vmin), 0.001, 1.0)
    hz = tf.math.add(tf.math.multiply(img, bw), vmin)        
    return hz

def param2(img):
    nbatch = K.shape(img)[0]
    Boo = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    bw = tf.math.subtract(vmax, vmin)
    gbw = tf.math.divide_no_nan((1 - bw), bw)
    gd = tf.math.multiply(tf.math.subtract(vmax, img), tf.math.subtract(img, vmin))
    gd = tf.math.multiply(gd, gbw)
    gdb = tf.math.multiply(tf.math.subtract(vmax, img), tf.math.subtract(Boo, vmin))
    gdb = tf.math.multiply(gdb, gbw)
    exp = tf.math.exp(-gd)
    expb = tf.math.exp(-gdb)
    return exp, expb, Boo
    
def param_comp(img):
    nbatch = K.shape(img)[0]
    Boo = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    bw = tf.math.subtract(vmax, vmin)
    gbw = tf.math.divide_no_nan((1 - bw), bw)    
    gdb = tf.math.multiply(tf.math.subtract(vmax, img), tf.math.subtract(Boo, vmin))
    gdb = tf.math.multiply(gdb, gbw)
    expb = tf.math.exp(-gdb)
    return expb, Boo
    
def cena(img):
    vmax, vmin, bw, vmed, vmdn = vals(img)
    ic = tf.math.subtract(img, vmin)
    ic = K.clip(tf.math.divide_no_nan(ic, bw), 0., 1.0)
    return ic

# ======================== ATTENTION MODULE ====================================
def topx(img):
    vmax, vmin, bw, vmed, vmdn = vals(img)
    outlier = tf.math.divide(K.clip(vmed-vmdn, 0., 1.0), vmed)
    thr = tf.math.subtract(1.0, outlier)
    imgx = K.clip(tf.math.subtract(img, tf.math.multiply(vmax, thr)), 0., 1.0)
    gr = tf.math.add(1.0, outlier)  #tf.math.divide(K.clip(vmed-vmdn, 0., 1.0), vmed))
    imgx_out = tf.math.divide_no_nan(imgx, img)
    unb = K.abs(tf.math.subtract(vmed, tf.reduce_mean(vmed)))
    imgx_out = tf.math.multiply(unb, tf.math.multiply(gr, imgx_out))
    return imgx_out
# =============================================================================

# ============================================================================
def ic_idb2(img):
    
    vmax, vmin, bw, vmed, B = vals(img)
    exp, expb, Boo = param2((img))    
    
    ic = cena(img)
    imgat = cena(topx(img))              # <<< uncomment to use the attention module
    
    ic = tf.math.subtract(ic, tf.math.multiply(B, (1-expb)))
    ic = K.clip(tf.math.divide_no_nan(ic, exp), 0., 1.0)   

    imgb0 = tf.math.multiply(img, exp)
    imgb1 = tf.math.multiply((1.0 - expb), Boo)
    imgb = tf.math.add(imgb0, imgb1)
    
    expn, expbn, Bn = param2(imgb)
    
    b0 = tf.math.multiply(imgb, expn)
    b1 = tf.math.multiply((1.0 - expbn), Bn) 
    idb = tf.math.add(b0, b1)
    
    idb = tf.math.add(imgat, idb)   # <<< uncomment to use the attention module
    
    return ic, idb

# ============================================================================

# ========================== Components (Loss sc) ============================
def comp(img, teta):
    expb, Boo = param_comp(img)
    compB = tf.math.multiply(teta, tf.math.multiply((1.0 - expb), Boo))
    compJ = tf.math.subtract(img, compB)
    return compJ, compB

# ============================================================================
#============================ Degradation Function ===========================
def dblock(img0, img1):

    imgc, imgdb = ic_idb2(img0)
    img_map = tf.math.subtract(imgdb, imgc)
    img_out = K.clip(tf.math.add(img1, img_map), 0.001, 1.0)
    
    return img_out
##============================================================================
# ================================== LOSS ====================================

def loss_RGB(y_true, y_pred):

    eta = 0.       ##
    teta = tf.constant([1.0 - eta], dtype=float)    
    
    ka = tf.constant([0.65], dtype=float)      
    kb = tf.constant([0.35], dtype=float)  

    y_pred = dblock(y_true, y_pred)
    
    compJt, compBt = comp(y_true, teta)
    compJp, compBp = comp(y_pred, teta)
    
    loss_mse_rgb = K.mean(K.square(y_true - y_pred))
    loss_compJ_rgb = K.mean(K.square(compJt - compJp))

    lossp = ka*loss_compJ_rgb + kb*loss_mse_rgb

    return lossp

# ============================================================================
def faz_DB(img):    
    img1 = dblock(img[0], img[1])
    return img1
# ============================================================================

#======================= CALLBACK EARLYSTOPPING===============================
class new_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        lim = 10e-4
        if(logs.get('val_loss') < lim):
            rede_cDB.stop_training = True
            print("\n Loss de validacao menor que", lim, " - fim do treinamento") 
        return

callbacks = new_callback()
#=============================================================================

outputDB_img = Lambda(faz_DB, name='SDB')([input_img, output_img])

rede_cDB = Model(input_img, output_img, name='rede')
rede_cDB.compile(optimizer=opt, loss=loss_RGB)

DB = Model(input_img, outputDB_img)
DB.compile(optimizer=opt, loss=loss_RGB)

# ===========================================================================

# ===========================================================================

epochs = 200
print("")
print("Training the model:")
print('Epochs = ', epochs, ' e Batch-size = ', batch_size)

H0 = rede_cDB.fit(data_train, data_train,      
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[callbacks],
              validation_split=0.1,
              steps_per_epoch= None,
              validation_steps=None)

esp_saida_rede_cDB = rede_cDB.predict(data_test)
saida_DB = DB.predict(data_test)

# agora = datetime.now()
# hora_fim = agora.strftime("%H:%M:%S")
# print("Start: ", hora_inicio)
# print("End: ", hora_fim)

# ============================================================================

# ============================================================================


# N = np.arange(0, epochs)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H0.history["loss"], label="Train_loss")
# plt.plot(N, H0.history["val_loss"], label="Val_loss")
# plt.title("")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(loc="upper right")
# plt.savefig('PATH')
# plt.show()


# ============================================================================
# Show samples of the test images at the end of training
    
qx = -2
plt.style.use('default')
for i in range(1, 5):

    plt.subplot(3, 4, i)
    p1 = plt.imshow(seleta[i+qx])
    plt.title('Input image', fontsize=8)
    plt.axis('off')
    p1.axes.get_xaxis().set_visible(False)
    p1.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 4, i+4)
    p2 = plt.imshow(esp_saida_rede_cDB[i+qx])
    plt.title('Output image', fontsize=8)    
    plt.axis('off')
    p2.axes.get_xaxis().set_visible(False)
    p2.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 4, i+8)
    p3 = plt.imshow(saida_DB[i+qx])
    plt.title('Output DF', fontsize=8)    #DF => Degradation Function
    plt.axis('off')
    p3.axes.get_xaxis().set_visible(False)
    p3.axes.get_yaxis().set_visible(False)



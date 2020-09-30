import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image


def img_to_np(path, resize = True):  
    img_array = []
    fpaths = glob.glob(path)
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images



def detector_fit(path):
    path += '/*.*'
    train = img_to_np(path)
    train = train.astype('float32') / 255.
    print(train.shape)
    
    encoding_dim = 1024    
    encoder_net = tf.keras.Sequential(
      [
          InputLayer(input_shape=train[0].shape),
          Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
          Flatten(),
          Dense(encoding_dim,)
      ])
    
    decoder_net = tf.keras.Sequential(
      [
          InputLayer(input_shape=(encoding_dim,)),
          Dense(8*8*128),
          Reshape(target_shape=(8, 8, 128)),
          Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
      ])
    
    
    # initialize outlier detector
    od = OutlierAE( threshold = 0.001,
                    encoder_net=encoder_net,
                    decoder_net=decoder_net,)
    
    adam = tf.keras.optimizers.Adam(lr=1e-4)
    
    # train
    od.fit(train, epochs=100, verbose=True,
           optimizer = adam)
    
    preds = od.predict(train, outlier_type='instance',
                return_instance_score=True,
                return_feature_score=True)
    
    return preds

import pandas as pd
import numpy as np
from PIL import Image
import os
import shutil
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, \
     Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image

def get_file_paths(path):
    file_paths = []
    for p in os.listdir(path):
        full_path = os.path.join(path, p)
        
        if (len(full_path) > 260) and (full_path[0] != "\\"):
            full_path = "\\\\?\\" + full_path
        
        if os.path.isfile(full_path):
            file_paths.append(full_path)
    
    return file_paths


def img_to_np(path, resize = True):  
    img_array = []
    fpaths = get_file_paths(path)
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): img = img.resize((32,32))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images

def detector_fit(path):
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
          Dense(4*4*128),
          Reshape(target_shape=(4, 4, 128)),
          Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
      ])
    
    
    # initialize outlier detector
    od = OutlierAE( threshold = 0.001,
                    encoder_net=encoder_net,
                    decoder_net=decoder_net)
    
    adam = tf.keras.optimizers.Adam(lr=1e-4)
    
    # train
    od.fit(train, epochs=150, verbose=True,
           optimizer = adam)
    
    preds = od.predict(train, outlier_type='instance',
                return_instance_score=True,
                return_feature_score=True)
    
    return preds

def write_output(preds, input_path, output_path):
    file_paths = get_file_paths(input_path)
    
    #writing images
    for i, path in enumerate(file_paths):
        if(preds['data']['is_outlier'][i] == 1):
            source = path
            shutil.copy(source, output_path)
    
    #writing CSV files
    file_names = [os.path.basename(item) for item in file_paths]

    dict1 = {'Filename': file_names,
     'instance_score': preds['data']['instance_score'],
     'is_outlier': preds['data']['is_outlier']}
     
    df = pd.DataFrame(dict1)
    df = df[df['is_outlier'] == 1]
    df.to_csv(output_path + '\\outliers.csv', index=False)
    
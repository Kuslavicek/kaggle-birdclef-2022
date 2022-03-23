import pandas as pd
import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout, TimeDistributed, Flatten
import matplotlib.pyplot as plt
from  tensorflow.keras.applications.vgg16 import VGG16


train_df = pd.read_csv() #Path to df2.csv
birds_id = pd.read_csv() #Path to birds_id.csv
train_df.sample(frac=1).reset_index(drop=True)
"""Creating train and validation set"""
valid_df = train_df[10000:].reset_index(drop=True)
train_df = train_df[:10000]
print(train_df.shape)
print(valid_df.head())

"""Generator loading spectrogram and id"""
class VoiceGenerator(tf.keras.utils.Sequence):
    def __init__(self,df,id_list):
        self.df = df
        self.id_list = id_list
    def __getitem__(self,index):
        X = self.__generate_X(index)
        y = self.__generate_y(index)
        return X, y
    def __len__(self):
        return int(len(self.df))
    def __generate_X(self,index):
        row = self.df.loc[index,:]
        file = "train_spec_128/"+row['filename'][:-4]+".npy"
        array = np.load(file)
        return np.resize(array,(1,128,128,1))        
    def __generate_y(self,index):
        array = np.zeros(21,np.int8)
        row = self.df.loc[index,:]
        id = row['primary_label']
        number = self.id_list.loc[self.id_list['string']==id,'index']
        array[number]=1
        return np.resize(array,(1,21))
    
    
    
"""Custom convolutional model"""
def build_conv():
    conv_model = keras.Sequential()
    conv_model.add(Conv2D(64,1,1,activation='relu'))
    conv_model.add(Conv2D(64,1,1,activation='relu'))
    conv_model.add(Conv2D(64,1,1,activation='relu'))
    conv_model.add(MaxPooling2D(2,2))
    conv_model.add(Conv2D(128,2,activation='relu'))
    conv_model.add(Conv2D(128,2,activation='relu'))
    conv_model.add(Conv2D(128,2,activation='relu'))
    conv_model.add(MaxPooling2D(2,2))
    conv_model.add(Dropout(0.5))
    conv_model.add(Dense(64, activation='relu'))
    conv_model.add(Dense(64, activation='relu'))
    conv_model.add(Dense(64, activation='relu'))
    conv_model.add(Flatten())
    conv_model.add(Dense(21,activation='softmax'))
    conv_model.compile(optimizer = 'Adam',loss= 'categorical_crossentropy',metrics = ['accuracy'] )
    return conv_model


"""Fitting to VGG16 model"""

model = VGG16(classes=21,weights=None, input_shape=(128,128,1))
model.compile(optimizer = 'Adam',loss= 'categorical_crossentropy',metrics = ['accuracy'] )
history=model.fit(VoiceGenerator(train_df,birds_id),epochs=3)
model.evaluate(VoiceGenerator(valid_df,birds_id))
model.summary()


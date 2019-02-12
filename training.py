
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from keras.utils import Sequence
from keras import layers
from keras import models
from keras.layers import Conv2D,BatchNormalization,Dense,Dropout,Flatten,MaxPooling2D,GlobalAveragePooling2D
from keras.models import Model

from google.colab import drive
drive.mount("/content/drive")

import os

data=pd.read_csv("drive/My Drive/training.csv")
print(data.head())
dataset=data.sample(frac=1).reset_index(drop=1)
print(dataset.head())
a=dataset["image_name"]
print(a.head())




input_shape=(240,320,3)
model = my_customisable_XceptionNet(input_shape=input_shape,number_of_inner_blocks=1)
model shown in different file

model.compile(optimizer='adam',loss="mse",metrics=["accuracy"])

model.summary()



X_paths=["drive/My Drive/efgh_320_240/"+i for i in a]
Y=dataset.drop(columns=["image_name"],axis=1)
Y=np.array(Y,dtype="float32")

Y[:,:2]=Y[:,:2]/2
Y[:,2:]=Y[:,2:]/2

print(Y[:5,:])
print(a.head())
print(X_paths[:5])

X_paths=np.array(X_paths, dtype='float32')


splitting into 13200:800 ratio

X_path_train = X_paths[:13200]
Y_train = Y[:13200,:]
X_path_val = X_paths[13200:]
Y_val = Y[13200:,:]


training_generator = Generator(X_path_train ,Y_train, 16)
val_generator = ValGenerator(X_path_val ,Y_val,16)

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

model_checkpoint = ModelCheckpoint("/drive/My Drive/ weights/my_weights_12_try.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

history = model.fit_generator(generator=training_generator,
                    use_multiprocessing=True ,
                    validation_data=val_generator,
                    workers=4,
                    callbacks= [model_checkpoint],
                    epochs=15)




model.save("drive/My Drive/models/exception_customised_64layers.h5")





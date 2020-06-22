import numpy as np
import random
from preprocessing import loadImages, resizing, grayReduction, imagesStandarization

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.metrics import accuracy

model = Sequential()
model.add(Dense(10, activation='relu',input_dim=16384))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))

train_images = np.load('saved_images/images_array_standar.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])
model.fit(x,y,batch_size=100,epochs=100)

images = loadImages("chest_xray","val","NORMAL")
images2 = loadImages("chest_xray","val","PNEUMONIA")
images = resizing(images, (128,128))
images2 = resizing(images2, (128,128))
images = grayReduction(images)
images2 = grayReduction(images2)
images = imagesStandarization(images)
images2 = imagesStandarization(images2)
y = np.concatenate((np.zeros(len(images)),np.ones(len(images2))))
x = images + images2
x = np.array(x).reshape((16,16384))
y = np.array(y)

z = model.evaluate(x,y)

print(z)


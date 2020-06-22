import numpy as np
import random
from preprocessing import loadImages, resizing, grayReduction, imagesStandarization, imagesNormalization, imagesToArray

images_normal = loadImages("chest_xray","train","NORMAL")
images_pneumonia = loadImages("chest_xray","train","PNEUMONIA")

images_normal = resizing(images_normal, (128,128))
images_pneumonia = resizing(images_pneumonia, (128,128))

images_normal = grayReduction(images_normal)
images_pneumonia = grayReduction(images_pneumonia)

#Estandarización
images_normal_standar = imagesStandarization(images_normal)
images_pneumonia_standar = imagesStandarization(images_pneumonia)

y_standar = np.concatenate((np.zeros(len(images_normal_standar)),np.ones(len(images_pneumonia_standar))))
x_standar = images_normal_standar + images_pneumonia_standar

temp = list(zip(x_standar,y_standar))
random.shuffle(temp)
x_standar,y_standar = zip(*temp)
x_standar = list(x_standar)
y_standar = list(y_standar)

x_standar = np.array(x_standar).reshape((5216,16384))
y_standar = np.array(y_standar).reshape((5216,1))

dump_array = np.concatenate((x_standar,y_standar),axis=1)
np.save('saved_images/images_array_standar.npy', dump_array)

#Normalización
images_normal_normal = imagesNormalization(images_normal)
images_pneumonia_normal = imagesNormalization(images_pneumonia)

y_normal = np.concatenate((np.zeros(len(images_normal_normal)),np.ones(len(images_pneumonia_normal))))
x_normal = images_normal_normal + images_pneumonia_normal

temp = list(zip(x_normal,y_normal))
random.shuffle(temp)
x_normal,y_normal = zip(*temp)
x_normal = list(x_normal)
y_normal = list(y_normal)

x_normal = np.array(x_normal).reshape((5216,16384))
y_normal = np.array(y_normal).reshape((5216,1))

dump_array = np.concatenate((x_normal,y_normal),axis=1)
np.save('saved_images/images_array_normal.npy', dump_array)

#Sin edición
images_normal = imagesToArray(images_normal)
images_pneumonia = imagesToArray(images_pneumonia)

y = np.concatenate((np.zeros(len(images_normal)),np.ones(len(images_pneumonia))))
x = images_normal + images_pneumonia

temp = list(zip(x,y))
random.shuffle(temp)
x,y = zip(*temp)
x = list(x)
y = list(y)

x = np.array(x).reshape((5216,16384))
y = np.array(y).reshape((5216,1))

dump_array = np.concatenate((x,y),axis=1)
np.save('saved_images/images_array.npy', dump_array)

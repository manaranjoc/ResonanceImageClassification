import numpy as np
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.metrics import accuracy

import time

model = Sequential()
model.add(Dense(10, activation='relu',input_dim=16384))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))

train_images = np.load('../saved_images/images_array_standar.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

kf = KFold(n_splits=10)
kf.get_n_splits(x)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])

exactitud = 0

start = time.time()

for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train,y_train,batch_size=100,epochs=100)

    exactitud += model.evaluate(X_test,y_test, verbose=0)[1]

error_promedio = 1-(exactitud/10)

print('Error para red: ',error_promedio)

elapsed_time = time.time()-start

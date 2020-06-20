import numpy as np
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.metrics import accuracy

from joblib import load


import time

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from metrics import metrics, meanMetrics, printMetrics, stdMetrics

model = Sequential()
model.add(Dense(10, activation='relu',input_dim=16384))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

train_images = np.load('saved_images/images_array_normal.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

feature_model = load('feature_extraction.joblib')
x = feature_model.transform(x)

num_splits = 10

kf = KFold(n_splits=num_splits)
kf.get_n_splits(x)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])

exactitud = np.zeros((num_splits, 5))
iteration = 0

start = time.time()

for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train,y_train,batch_size=100,epochs=5)

    y_pred = model.predict_classes(X_test)

    exactitud[iteration,:] = metrics(y_test, y_pred)
    iteration += 1


elapsed_time = time.time()-start

mean_metrics = meanMetrics(exactitud)
std_metrics = stdMetrics(exactitud)
printMetrics(mean_metrics)
print('Desviación estandar')
print('########################################')
printMetrics(std_metrics)

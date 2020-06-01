from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np

import matplotlib.pyplot as plt

import time

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from metrics import metrics, meanMetrics, printMetrics

train_images = np.load('../saved_images/images_array_normal.npy')

x = train_images[:,:-1]
y = train_images[:,-1]

print("Images already Loaded")

num_splits = 10

kf = KFold(n_splits=num_splits)
kf.get_n_splits(x)

k = 10

error_by_k = np.zeros((k, 5))

start = time.time()

for i in range(1,k+1):
    clf = KNeighborsClassifier(n_neighbors=i, weights='uniform',n_jobs=-1)

    error_promedio = np.zeros((num_splits,5))
    iteration = 0

    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        error_promedio[iteration, :] = metrics(y_test, y_pred)
        iteration += 1

    error_promedio = meanMetrics(error_promedio)

    print('Error para', i, ' vecinos: ')
    print('########################################')
    printMetrics(error_promedio)

    error_by_k[i-1, :]=error_promedio


elapsed_time = time.time()-start

plt.plot(range(1,k+1), error_by_k[:,0], 'b--')
plt.xlabel('Numero de vecinos')
plt.ylabel('Error asociado')
plt.show()
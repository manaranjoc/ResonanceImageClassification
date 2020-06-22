from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import numpy as np

from joblib import load

import time

import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from metrics import metrics, meanMetrics, printMetrics, stdMetrics


train_images = np.load('../saved_images/images_array_normal.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

feature_model = load('feature_extraction')
x = feature_model.transform(x)

#C = np.logspace(1,2,2)
C = [0.01]
#Gamma = np.logspace(-3,2,6)

num_splits = 10

kf = KFold(n_splits=num_splits)
kf.get_n_splits(x)

error_by_parameter = np.zeros((6,5))
i = 0

start = time.time()

for c in C:
    clf = LinearSVC(C=c, max_iter = 100000)

    exactitud = np.zeros((num_splits, 5))
    iteration = 0

    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        exactitud[iteration, :] = metrics(y_test, y_pred)
        iteration += 1

    error_standard = stdMetrics(error_promedio)
    error_promedio = meanMetrics(error_promedio)

    print('Error para C=', c)
    printMetrics(error_promedio)
    print('Desviación estandar')
    print('###################################')
    printMetrics(error_standard)

    error_by_parameter[i,:]=error_promedio
    i += 1


elapsed_time = time.time()-start
print('Elapsed time for one neuron Classification: ',elapsed_time)

plt.plot(C, error_by_parameter[:,0], 'b--')


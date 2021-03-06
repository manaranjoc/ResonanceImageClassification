from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

from joblib import load

import matplotlib.pyplot as plt


import time

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from metrics import metrics, meanMetrics, printMetrics, stdMetrics

train_images = np.load('saved_images/images_array_normal.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

print("Images already Loaded")

#feature_model = load('wrapping_method.joblib')
#x = feature_model.transform(x)

num_splits = 10

kf = KFold(n_splits=num_splits)
kf.get_n_splits(x)

error_by_B = np.zeros((5,5))

i = 0

start = time.time()

saveFile = open('res.dat', 'w')

for B in [30]:
#for B in range(10,51,10):

    error_promedio = np.zeros((num_splits, 5))
    iteration = 0

    clf = RandomForestClassifier(n_estimators=B, n_jobs=-1)

    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        error_promedio[iteration, :] = metrics(y_test, y_pred)
        iteration += 1

    error_desviacion = stdMetrics(error_promedio)
    error_promedio = meanMetrics(error_promedio)

    print('Error para', B, ' forest: ')
    print('###########################################')
    printMetrics(error_promedio)
    print('Desviación estandar')
    print('##########################################')
    printMetrics(error_desviacion)

    np.savetxt(saveFile, error_promedio)

    error_by_B[i, :]=error_promedio
    i+=1

saveFile.close()
elapsed_time = time.time()-start
print('Elapsed time for one neuron Classification: ',elapsed_time)

plt.plot(range(10,51,10), error_by_B[:, 0], 'b--')
plt.xlabel('Numero de Arboles')
plt.ylabel('Error asociado')
plt.show()




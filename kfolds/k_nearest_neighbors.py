from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np

import matplotlib.pyplot as plt

import time

train_images = np.load('saved_images/images_array_normal.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

print("Images already Loaded")

kf = KFold(n_splits=10)
kf.get_n_splits(x)

k = 10

error_by_k = np.zeros(k)

start = time.time()


for i in range(1,k+1):
    clf = KNeighborsClassifier(n_neighbors=i, weights='uniform',n_jobs=-1)
    exactitud = 0
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train,y_train)

        exactitud += clf.score(X_test, y_test)

    error_promedio = 1-(exactitud/10)

    print('Error para', i, ' vecinos: ',error_promedio)

    error_by_k[i-1]=error_promedio

elapsed_time = time.time()-start

plt.plot(range(1,k+1), error_by_k, 'b--')
plt.xlabel('Numero de vecinos')
plt.ylabel('Error asociado')
plt.show()
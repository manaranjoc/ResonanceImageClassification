from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import numpy as np

import time

train_images = np.load('../saved_images/images_array_normal.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

C = np.logspace(1,2,2)
Gamma = np.logspace(-3,2,6)

kf = KFold(n_splits=10)
kf.get_n_splits(x)

error_by_parameter = np.zeros((6,6))
i = 0

start = time.time()

for c in C:
        clf = LinearSVC(C=c, max_iter = 100000)
        exactitud = 0
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train,y_train)

            exactitud += clf.score(X_test, y_test)

        error_promedio = 1-(exactitud/10)

        print('Error para C=', c ,error_promedio)

        error_by_parameter[int(i/5),i%5]=error_promedio
        i += 1

elapsed_time = time.time()-start
print('Elapsed time for one neuron Classification: ',elapsed_time)

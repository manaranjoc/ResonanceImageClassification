from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np

import matplotlib.pyplot as plt

train_images = np.load('saved_images/images_array_standar.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

C = np.logspace(-3,2,5)
Gamma = np.logspace(-3,2,5)

kf = KFold(n_splits=10)
kf.get_n_splits(x)

error_by_parameter = np.zeros((5,5))
i = 0
for c in C:
    for gamma in Gamma:
        clf = SVC(kernel='rbf', C=c, gamma=gamma)
        exactitud = 0
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train,y_train)

            exactitud += clf.score(X_test, y_test)

        error_promedio = 1-(exactitud/10)

        print('Error para C=', c, ' y gamma=', gamma,': ',error_promedio)

        error_by_parameter[int(i/5),i%5]=error_promedio
        i += 1


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

import matplotlib.pyplot as plt

train_images = np.load('../saved_images/images_array_normal.npy')
x = train_images[:,:-1]
y = train_images[:,-1]

print("Images already Loaded")

kf = KFold(n_splits=10)
kf.get_n_splits(x)

error_by_B = np.zeros(5)

i = 0

for B in range(10,51,10):
    exactitud = 0
    clf = RandomForestClassifier(n_estimators=B)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train,y_train)

        exactitud += clf.score(X_test, y_test)

    error_promedio = 1-(exactitud/10)

    print('Error para', B, ' forest: ',error_promedio)

    error_by_B[i]=error_promedio
    i+=1

plt.plot(range(10,51,10), error_by_B, 'b--')
plt.xlabel('Numero de Arboles')
plt.ylabel('Error asociado')
plt.show()




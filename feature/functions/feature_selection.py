import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import KFold

from metrics import metrics

import time

def select_features(classifier, n_features, fwd, fltg):

    sfs = SFS(classifier,
        k_features=n_features,
        forward=fwd,
        floating=fltg,
        verbose=1,
        scoring='accuracy',
        cv=10,
        n_jobs=-1)

    return sfs

def select_features_percentage(classifier, percentage, fwd, fltg, X, Y):
    tiempo_i = time.time()

    Errores = np.ones(10)
    j = 0
    kf = KFold(n_splits=10)
    clf = classifier

    sf = select_features(clf, int(X.shape[0]*10/100), fwd, fltg)

    sf = sf.fit(X, Y)

    X_sf = sf.transform(X)

    for train_index, test_index in kf.split(X_sf):

        X_train, X_test = X_sf[train_index], X_sf[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Errores[j] = 1-metrics(y_test,y_pred)[0]
        j+=1

    print("\nError de validación aplicando SFS: "+str(np.mean(Errores))+"+/-"+str(np.std(Errores)))
    print("\nEficiencia en validación aplicando SFS: "+str((1-np.mean(Errores))*100)+"%")
    print("\nTiempo total de ejecución: "+str(time.time()-tiempo_i)+" segundos.")

    return sf


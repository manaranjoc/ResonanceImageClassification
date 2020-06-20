import numpy as np
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as LDA
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.model_selection import KFold

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from metrics import metrics, meanMetrics, printMetrics, stdMetrics

import time

def extract_features(extraction_type, n_components):

    if extraction_type == 'pca':
        ext = PCA(n_components=n_components)

        return ext
    elif extraction_type == 'lda':
        ext = LDA(n_discriminants=n_components)

        return ext
    
    else:
        print("Input a valid method for (PCA or LDA)\n")
        

def extract_features_percentage(classifier, percentage, X, Y, extraction_type):
    tiempo_i = time.time()

    Errores = np.ones(10)
    Metrics = np.zeros((10,5))
    j = 0
    kf = KFold(n_splits=10)
    clf = classifier

    ex = extract_features(extraction_type, int(X.shape[1]*percentage/100))
    
    if(extraction_type == "pca"):
        ex = ex.fit(X)
    elif(extraction_type == "lda"):
        ex = ex.fit(X,Y.astype(int))

    X_ex = ex.transform(X)

    for train_index, test_index in kf.split(X_ex):

        X_train, X_test = X_ex[train_index], X_ex[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        Errores[j] = 1-metrics(y_test, y_pred)[0]
        Metrics[j,:] = metrics(y_test, y_pred)
        j+=1

    print("\nError de validación aplicando "+str(extraction_type)+" at "+str(percentage)+"%: "+str(np.mean(Errores))+"+/-"+str(np.std(Errores)))
    print("\nEficiencia en validación aplicando "+str(extraction_type)+" at "+str(percentage)+"%: "+str((1-np.mean(Errores))*100)+"%")
    print("\nTiempo total de ejecución: "+str(time.time()-tiempo_i)+" segundos.")
    
    MetricsMean = meanMetrics(Metrics)
    MetricsStd = stdMetrics(Metrics)

    printMetrics(MetricsMean)
    print("\nDesviaciones Estandard")
    printMetrics(MetricsStd)

    return ex


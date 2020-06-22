import functions.feature_extraction as fe
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

train_images = np.load('saved_images/images_array_normal.npy')

x = train_images[:,:-1]
y = train_images[:,-1]
print(x.shape)

print("Images already Loaded")

clf = KNeighborsClassifier(n_neighbors=8, weights='uniform',n_jobs=-1)

feature_extraction = fe.extract_features_percentage(clf, 25, x, y, 'pca')

dump(feature_extraction, 'feature_extraction.joblib')
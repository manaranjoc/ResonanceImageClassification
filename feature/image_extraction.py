import functions.feature_extraction as fe
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_images = np.load('saved_images/images_array_normal.npy')

x = train_images[:,:-1]
y = train_images[:,-1]
print(x.shape)

print("Images already Loaded")

clf = KNeighborsClassifier(n_neighbors=8, weights='uniform',n_jobs=-1)

print(fe.extract_features_percentage(clf, 25, x, y, 'lda').transform(x).shape)
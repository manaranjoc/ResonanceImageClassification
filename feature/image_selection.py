import functions.feature_selection as fs
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from joblib import dump

train_images = np.load('saved_images/images_array_normal.npy')

x = train_images[:,:-1]
y = train_images[:,-1]
print(x.shape)

print("Images already Loaded")

clf = KNeighborsClassifier(n_neighbors=8, weights='uniform')

# Feature Selection by wrapping method
#1. SFS True False
#2. SBS False False
#3. SFFS True True
#4. SBFS False True
wrapping_method = fs.select_features_number(clf, 15, True, False, x, y)

dump(wrapping_method, 'wrapping_method.joblib')

# Feature Selection by filter method
#filter_method = fs.select_features_filter_percentage(clf, 75, x, y)
#print(filter_method.transform(x).shape)
#print(filter_method.get_params())
#print(filter_method.get_support())

#dump(filter_method, 'filter_method.joblib')

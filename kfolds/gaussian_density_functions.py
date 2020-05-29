from sklearn.model_selection import KFold
import numpy as np

train_images = np.load('saved_images/images_array.npy')

class_filter = train_images[:,-1]==1

x_1 = train_images[class_filter,:-1]
x_0 = train_images[np.logical_not(class_filter),:-1]

mu_c0 = np.mean(x_0, 0)
mu_c1 = np.mean(x_1, 0)

sigma0 = np.cov(x_0.T)
sigma1 = np.cov(x_1.T)

det_s0 = np.linalg.det(sigma0)
det_s1 = np.linalg.det(sigma1)

#inv_s0 = np.linalg.inv(sigma0)
#inv_s1 = np.linalg.inv(sigma1)

print(np.sqrt(det_s0))
print(np.sqrt(det_s1))
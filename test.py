import math
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt

from data_processing.DIS_dataset import DIS_dataset
from models.knn import KNN

dataset = DIS_dataset(data_folder="2_spheres_2.0-4.0mm", mcml_flag=False)

X_train, X_test, Y_train, Y_test = dataset.get_scipy_data()
knn = KNN()
knn.fit(dataset)
#knn.model = pickle.load(open('knn_HQ', 'rb'))
pickle.dump(knn.model, open("models/scipy_models/knn_HQ", 'wb'))
X_pred = knn.model.predict(Y_test)
mua = X_test[:, 0]
mus = X_test[:, 1]
mua_pred = X_pred[:, 0]
mus_pred = X_pred[:, 1]
mua_error = np.abs(np.divide(mua_pred - mua, mua+1e-6))
mus_error = np.abs(np.divide(mus_pred - mus, mus+1e-6))
print(np.median(mua_error))
mua_error = mua_error[mua_error < 1]
plt.hist(mua_error, bins=30)
plt.show()
print(np.median(mus_error))
mus_error = mus_error[mus_error < 1]
plt.hist(mus_error, bins=30)
plt.show()

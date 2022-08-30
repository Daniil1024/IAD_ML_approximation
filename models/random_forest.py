import math
import os
import pickle

import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from data_processing.DIS_dataset import DIS_dataset
class RandomForest():
    def fit(self, dataset, include_thickness=True):
        X_train, X_test, Y_train, Y_test = dataset.get_scipy_data(include_thickness=include_thickness)
        self.model.fit(Y_train, X_train)
        X_pred = self.model.predict(Y_test)

        mua = X_pred[:, 0]
        mus = X_pred[:, 1]
        # thickness = X_pred[:, 2]
        mua_uncertainty = np.median(np.abs(np.divide(mua - (X_test[:, 0]), (X_test[:, 0]))))
        mus_uncertainty = np.median(np.abs(np.divide(mus - (X_test[:, 1]), (X_test[:, 1]))))
        # thickness_uncertainty = np.abs(np.divide(mua - X_test[:, 2], X_test[:, 2])).mean()
        print(
            "mua unc: ", str(mua_uncertainty),
            "mus unc: ", str(mus_uncertainty))
        print("R^2 = ", r2_score(X_test, X_pred))
        print("MSE = ", math.sqrt(mean_squared_error(X_test, X_pred)))

    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))

    def __init__(self, n_estimators=100, max_depth=10):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

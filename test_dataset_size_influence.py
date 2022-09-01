import math
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from data_processing.DIS_dataset import DIS_dataset
from models.knn import KNN
from models.random_forest import RandomForest
def visualise_history(history):
    points = np.linspace(.5, 6.5, 13)
    arr = np.array(pickle.load(open("history", 'rb')))
    plt.title("Fractional error of KNN vs training set size", fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("Number of data points / $10^5$", fontsize=24)
    plt.ylabel("Median fractional error on the test set", fontsize=24)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3))
    plt.plot(points, arr[:, 1])
    plt.show()

my_dataset = DIS_dataset(data_folder="2_spheres_2.0-4.0mm", mcml_flag=False)
knn = KNN()
#forest.load("models/scipy_saved_models/knn_HQ")
history = []
scale_thickness = 100
points = np.linspace(50000, 650000, 13)
last_N_points = 160000
test_data = np.concatenate([my_dataset.data["mua_mean"].reshape(-1, 1)[-last_N_points:],
                            my_dataset.data["mus_mean"].reshape(-1, 1)[-last_N_points:],
                            my_dataset.data["thickness"][my_dataset.successful_indices].reshape(-1, 1)[-last_N_points:] / scale_thickness, # the scaling factor for KNN is 100 (magic number)
                          my_dataset.data["M_R"].reshape(-1, 1)[-last_N_points:],
                          my_dataset.data["M_T"].reshape(-1, 1)[-last_N_points:],
                          my_dataset.data["thickness"][my_dataset.successful_indices].reshape(-1, 1)[-last_N_points:] / scale_thickness], axis = 1)
X_test = test_data[:, :3]
Y_test = test_data[:, 3:]
for fraction in points:
    fraction = int(np.floor(fraction))
    print(fraction)
    data = np.concatenate([my_dataset.data["mua_mean"].reshape(-1, 1)[:fraction],
                            my_dataset.data["mus_mean"].reshape(-1, 1)[:fraction],
                            my_dataset.data["thickness"][my_dataset.successful_indices].reshape(-1, 1)[:fraction] / scale_thickness, # the scaling factor for KNN is 100 (magic number)
                          my_dataset.data["M_R"].reshape(-1, 1)[:fraction],
                          my_dataset.data["M_T"].reshape(-1, 1)[:fraction],
                          my_dataset.data["thickness"][my_dataset.successful_indices].reshape(-1, 1)[:fraction] / scale_thickness], axis = 1)
    N = fraction
    X_train = data[:, :3]
    Y_train = data[:, 3:]
    knn.model.fit(Y_train, X_train)
    X_pred = knn.model.predict(Y_test)

    mua = X_pred[:, 0]
    mus = X_pred[:, 1]
    # thickness = X_pred[:, 2]
    mua_uncertainty = np.nanmedian(np.abs(np.divide(mua - (X_test[:, 0]), (X_test[:, 0]))))
    mus_uncertainty = np.nanmedian(np.abs(np.divide(mus - (X_test[:, 1]), (X_test[:, 1]))))
    # thickness_uncertainty = np.abs(np.divide(mua - X_test[:, 2], X_test[:, 2])).mean()
    print(
        "mua unc: ", str(mua_uncertainty),
        "mus unc: ", str(mus_uncertainty))
    print("R^2 = ", r2_score(X_test, X_pred))
    print("MSE = ", math.sqrt(mean_squared_error(X_test, X_pred)))
    history.append([fraction, mua_uncertainty, mus_uncertainty])
    print(mua_uncertainty, mus_uncertainty)
    print()
print("start predicting")
history = np.array(history)
pickle.dump(history, open("history", 'wb'))
visualise_history(history)



import pickle

import numpy as np

from data_processing.DIS_dataset import DIS_dataset
from models.knn import KNN
from models.random_forest import RandomForest

my_dataset = DIS_dataset(data_folder="2_spheres_2.0-4.0mm", mcml_flag=False)
knn = KNN()
#forest.load("models/scipy_saved_models/knn_HQ")
history = []
points = np.linspace(0, 1, 20)
for fraction in points:
    print(fraction)
        data = np.concatenate([self.data["mua_mean"].reshape(-1, 1),
                               self.data["mus_mean"].reshape(-1, 1),
                               self.data["thickness"][self.successful_indices].reshape(-1, 1) / scale_thickness,
                               # the scaling factor for KNN is 100 (magic number)
                               ],
                              axis=1)
    N = fraction
    train_size = int(N)
    test_size = N - train_size
    train, test = train_test_split(data, train_size=train_size, test_size=test_size)
    input_output_dimensions = 3 if include_thickness else 2
    X_train, X_test = train[:, :input_output_dimensions], test[:, :input_output_dimensions]
    Y_train, Y_test = train[:, input_output_dimensions:], test[:, input_output_dimensions:]
    mua_unc, mus_unc = knn.fit(my_dataset, train_fraction=fraction)
    history.append((fraction, mua_unc, mus_unc))
    print(mua_unc, mus_unc)
    print()
print("start predicting")
pickle.dump(history, open("history", 'wb'))

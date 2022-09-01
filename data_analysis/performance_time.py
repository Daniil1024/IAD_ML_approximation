import time
from data_processing.DIS_dataset import DIS_dataset
from ML.models.random_forest import RandomForest
my_dataset = DIS_dataset(data_folder="2_spheres_2.0-4.0mm", mcml_flag=False)
forest = RandomForest()
forest.load("ML/scipy_saved_models/RF_HQ")
X_train, X_test, Y_train, Y_test = my_dataset.get_scipy_data(include_thickness=True, scale_thickness=1)
print("start predicting")

start = time.process_time()

forest.model.predict(Y_train)

end = time.process_time()
print(end - start)
print(len(Y_train))

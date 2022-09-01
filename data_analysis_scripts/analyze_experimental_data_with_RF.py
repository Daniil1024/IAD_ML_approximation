import pickle

import matplotlib.pyplot as plt

from data_processing.dis_utils import read_dis_measurement_data
from data_processing.utils import read_rxt_file
import numpy as np
import glob
import os

from ML.models.KNN import KNN
from ML.models.random_forest import RandomForest

dataset_path = "../Data/DIS_processed/DIS/"

paths = glob.glob(dataset_path + "*")
paths = [path[:-2] for path in paths]
all_phantoms = list(dict.fromkeys(paths))

model = KNN()
model.load('ML/scipy_saved_models/knn_HQ')
thickness_scale = 100
#all_phantoms = ["Data/DIS_processed/DIS/T.1.1"]
for phantom_path in all_phantoms:
    PHANTOM = glob.glob(phantom_path + ".*")
    NAME = os.path.split(phantom_path)[-1]

    ASSUMED_ANISOTROPY = 0.7

    SAMPLES = []
    for _phantom in PHANTOM:
        SAMPLES = SAMPLES + glob.glob(_phantom + "/*/")

    print(NAME)
    print("\t", SAMPLES)

    measurements = []
    measurements_x = []
    measurements_mua = []
    measurements_mus = []
    measurement_success = []
    measurement_convergence_code = []
    num_samples = 0
    for folder_name in SAMPLES:
        _measurement = read_dis_measurement_data(folder_name, assumed_anisotropy=ASSUMED_ANISOTROPY)
        a, b, c, thickness = read_rxt_file(folder_name+'1.rxt')
        _measurement["thickness"] = thickness
        measurements.append(_measurement)
        measurements_x.append(_measurement["wavelength"])
        mua = _measurement["absorption"]
        mua[_measurement["success"] == False] = None
        measurements_mua.append(mua)
        mus = _measurement["scattering"]
        mus[_measurement["success"] == False] = None
        measurements_mus.append(mus)
        measurement_success.append(_measurement["success"])
        measurement_convergence_code.append(_measurement["convergence_code"])
        num_samples += 1

    measurement_success = np.any(np.reshape(np.asarray(measurement_success), (num_samples, -1)), axis=0)
    measurement_convergence_code = np.reshape(np.asarray(measurement_convergence_code), -1)
    all_measurements_x = measurements_x[0]
    measurements_x = np.nanmedian(np.reshape(np.asarray(measurements_x), (num_samples, -1)), axis=0)[measurement_success]
    mua_mean = np.nanmedian(np.reshape(np.asarray(measurements_mua), (num_samples, -1)), axis=0)[measurement_success]
    mus_mean = np.nanmedian(np.reshape(np.asarray(measurements_mus), (num_samples, -1)), axis=0)[measurement_success]

    measurements_model_mua = []
    measurements_model_mus = []
    for m_idx, measurement in enumerate(measurements):
        output_space = np.concatenate([measurement["reflectance"].reshape(-1, 1),
                                       measurement["transmittance"].reshape(-1, 1),
                                       measurement["thickness"]*np.ones_like(measurement["reflectance"].reshape(-1, 1))/thickness_scale], axis=1)
        prediction = model.model.predict(output_space)
        mua = prediction[:, 0]
        mus = prediction[:, 1]
        thickness_pred = prediction[:, 2]
        measurement["model_mua"] = mua
        measurement["model_mus"] = mus
        measurements_model_mua.append(mua)
        measurements_model_mus.append(mus)
    if NAME == "T.1.1":
        print(2)
    model_mua_mean = np.mean(np.reshape(np.asarray(measurements_model_mua), (num_samples, -1)), axis=0)
    model_mus_mean = np.mean(np.reshape(np.asarray(measurements_model_mus), (num_samples, -1)), axis=0)
    model_mua_std = np.std(np.reshape(np.asarray(measurements_model_mua), (num_samples, -1)), axis=0)
    model_mus_std = np.std(np.reshape(np.asarray(measurements_model_mus), (num_samples, -1)), axis=0)


    wavelength = measurements[0]["wavelength"]
    MIN_WL = min(measurements[0]["wavelength"])
    MAX_WL = max(measurements[0]["wavelength"])

    colours = plt.cm.get_cmap("viridis", len(measurements))

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"DIS characterisation of phantom material {NAME}")
    plt.subplot(1, 2, 1)
    plt.title("Optical scattering in sample")
    B_mus_exp, A_mus_exp = np.polyfit(all_measurements_x, np.log(model_mus_mean), deg=1)
    for m_idx, measurement in enumerate(measurements):
        plt.scatter(measurement["wavelength"],
                    measurement["model_mus"],
                    color=colours(m_idx), s=8, marker="*", alpha=0.2)
    x_range = np.arange(min(measurements_x), max(measurements_x))
    mus_interp = np.interp(x_range, all_measurements_x, model_mus_mean)
    mus_std_interp = np.interp(x_range, all_measurements_x, model_mus_std)
    plt.fill_between(x_range, mus_interp - mus_std_interp, mus_interp + mus_std_interp, color="red", alpha=0.3)
    plt.plot(x_range, mus_interp - mus_std_interp, color="darkred", alpha=1, linestyle="dashed", label="std")
    plt.plot(x_range, mus_interp + mus_std_interp, color="darkred", alpha=1, linestyle="dashed")
    plt.plot(x_range, mus_interp, label='B-Spline', color="green", linewidth=2)
    plt.plot(x_range, np.exp(A_mus_exp) * np.exp(B_mus_exp * x_range),
             label=f"y = {np.round(np.exp(A_mus_exp), 2)}e^{np.round(B_mus_exp, 5)}x",
             color="black", linewidth=2)

    plt.legend(loc="best")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Scattering [1/cm]")
    plt.legend(loc="best")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Scattering [1/cm]")

    plt.subplot(1, 2, 2)
    plt.title("Optical absorption in sample")
    for m_idx, measurement in enumerate(measurements):
        plt.scatter(measurement["wavelength"],
                    measurement["model_mua"],
                    color=colours(m_idx), s=8, marker="*", alpha=0.2)
    mua_interp = np.interp(x_range, all_measurements_x, model_mua_mean)
    mua_std_interp = np.interp(x_range, all_measurements_x, model_mua_std)
    plt.fill_between(x_range, mua_interp - mua_std_interp, mua_interp + mua_std_interp, color="red", alpha=0.3)
    plt.plot(x_range, mua_interp - mua_std_interp, color="darkred", alpha=1, linestyle="dashed", label="std")
    plt.plot(x_range, mua_interp + mua_std_interp, color="darkred", alpha=1, linestyle="dashed")
    plt.plot(x_range, mua_interp, label='B-Spline', color="green", linewidth=2)

    plt.legend(loc="best")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Absorption [1/cm]")
    plt.tight_layout()
    for _pname in PHANTOM:
        plt.savefig(_pname + "/" + "knn_" + NAME + ".png", dpi=300)
    plt.close()
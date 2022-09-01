from dis_utils import read_dis_measurement_data
import numpy as np
import glob
import os
#this script combines the information from rxt files to one npz file per sample

BASE_PATH = "../Data/DIS_processed/0_spheres_3.0mm/"

paths = glob.glob(BASE_PATH + "*")
paths = [path[:-2] for path in paths]  # we just strip the '.a' or '.b' each path will then be present twice (since stripping from T.1.1.a and T.1.1.b gives T.1.1
all_phantoms = list(dict.fromkeys(paths))  # we avoid redundancy by picking unique keys

for phantom_path in all_phantoms:
    PHANTOM = glob.glob(phantom_path + ".*")
    NAME = os.path.split(phantom_path)[-1]

    ASSUMED_ANISOTROPY = 0.7

    measurement_paths = []
    for _phantom in PHANTOM:
        measurement_paths = measurement_paths + (glob.glob(_phantom + "/*/"))
        # we cannot just use .append since glob.glob produces a list of strings rather than just a single string

    print(NAME)
    print("\t", measurement_paths)

    measurements = []
    measurements_x = []
    measurements_mua = []
    measurements_mus = []
    measurements_M_R = []
    measurements_M_T = []
    measurements_thickness = []
    measurement_success = []
    measurement_convergence_code = []
    num_measurements = 0
    for measurement_path in measurement_paths:
        try:
            _measurement = read_dis_measurement_data(measurement_path, assumed_anisotropy=ASSUMED_ANISOTROPY)
        except FileNotFoundError:
            continue
        measurements.append(_measurement)
        measurements_x.append(_measurement["wavelength"])
        mua = _measurement["absorption"]
        mua[_measurement["success"] == False] = None
        measurements_mua.append(mua)
        mus = _measurement["scattering"]
        mus[_measurement["success"] == False] = None
        measurements_mus.append(mus)
        measurements_M_R.append(_measurement["reflectance"])
        measurements_M_T.append(_measurement["transmittance"])
        measurements_thickness.append(_measurement["thickness"])
        measurement_success.append(_measurement["success"])
        measurement_convergence_code.append(_measurement["convergence_code"])
        num_measurements += 1
    if num_measurements == 0:
        continue
    # when averaging out we assume that across the measurements the number of points (and the wavelength values) were the same
    # for artificial data we usually have just one measurement for each sample so there is no averaging
    measurement_success = np.any(np.reshape(np.asarray(measurement_success), (num_measurements, -1)), axis=0)
    # NOTE: for experimental data with several measurements, convergence code does not make sense, since the values at each wavelength are averaged out
    measurement_convergence_code = np.reshape(np.asarray(measurement_convergence_code), -1)
    measurements_x = np.nanmedian(np.reshape(np.asarray(measurements_x), (num_measurements, -1)), axis=0)[measurement_success]
    mua_mean = np.nanmedian(np.reshape(np.asarray(measurements_mua), (num_measurements, -1)), axis=0)[measurement_success]
    mus_mean = np.nanmedian(np.reshape(np.asarray(measurements_mus), (num_measurements, -1)), axis=0)[measurement_success]
    mua_std = np.nanstd(np.reshape(np.asarray(measurements_mua), (num_measurements, -1)), axis=0)[measurement_success]
    mus_std = np.nanstd(np.reshape(np.asarray(measurements_mus), (num_measurements, -1)), axis=0)[measurement_success]
    M_R_mean_all = np.median(np.reshape(np.asarray(measurements_M_R), (num_measurements, -1)), axis=0)
    M_T_mean_all = np.median(np.reshape(np.asarray(measurements_M_T), (num_measurements, -1)), axis=0)
    thickness = np.mean(np.reshape(np.asarray(measurements_thickness), (num_measurements, -1)), axis=0)
    wavelength = measurements[0]["wavelength"]
    MIN_WL = min(measurements[0]["wavelength"])
    MAX_WL = max(measurements[0]["wavelength"])

    # y = Ae^Bx
    #    B_mus_exp, A_mus_exp = np.polyfit(measurements_x, np.log(mus_mean), deg=1)
    for _pname in PHANTOM:
        np.savez(os.path.join(_pname, NAME + ".npz"),
                 convergence_code=measurement_convergence_code,
                 successes=measurement_success,
                 wavelength=wavelength,  # save the wavelengths at the interpolation points
                 mua_mean=mua_mean,  # save the mean absorption for successful measurements
                 mua_std=mua_std,  # save the standard deviation of the absorption for successful measurements
                 mus_mean=mus_mean,  # save the mean scattering for successful measurements
                 mus_std=mus_std,  # save the standard deviation of the scattering for successful measurements
                 M_R_mean=M_R_mean_all,
                 M_T_mean=M_T_mean_all,
                 #       mus_fit=np.exp(A_mus_exp) * np.exp(B_mus_exp * wavelength), # save an exponential fit of the scattering
                 g=ASSUMED_ANISOTROPY*np.ones_like(mua_mean),
                 thickness=thickness)

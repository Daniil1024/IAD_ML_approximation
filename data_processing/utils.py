# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import linecache
import os
import numpy as np
import pandas as pd
import glob


def read_rxt_file(file_path: str) -> (np.ndarray, np.ndarray, np.ndarray, float):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"You need to supply the path to a file, not {file_path}")

    dataframe = pd.read_csv(filepath_or_buffer=file_path, delimiter=" ", skiprows=1, header=None)

    with open(file_path, "r") as metadata_path:
        metadata = metadata_path.readline().split("\t")

    return (np.asarray(dataframe[0].values), np.asarray(dataframe[1].values),
            np.asarray(dataframe[2].values), float(metadata[2]))


def read_reference_spectra(file_path: str) -> (np.ndarray, np.ndarray, float):  # no idea what this function is for
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"You need to supply the path to a file, not {file_path}")

    data = np.load(file_path)
    return data["mua_mean"], data["mus_mean"], data["g"]


def get_sample_data(
        sample_path: str) -> dict:  # TODO this function might duplicate the data points if we have '.a' and '.b' folders but I am not sure
    folder, name = os.path.split(sample_path)  # e.g. Data/DIS_processed/DIS and T.1.1.a
    short_name = name[:-2]  # e.g. T.1.1
    measurement_paths = glob.glob(os.path.join(os.path.join(folder, short_name + '.*'), '*', ''))
    path_npz = os.path.join(sample_path, short_name + ".npz")
    if not os.path.exists(path_npz):
        raise FileNotFoundError

    npz_data = np.load(path_npz, allow_pickle=True)
    successes = npz_data["successes"]
    convergence_code = npz_data["convergence_code"]
    wavelength = npz_data["wavelength"]
    mua_mean = npz_data["mua_mean"]
    mua_std = npz_data["mua_std"]
    mus_mean = npz_data["mus_mean"]
    mus_std = npz_data["mus_std"]
    M_R = npz_data["M_R_mean"]
    M_T = npz_data["M_T_mean"]
    thickness = npz_data["thickness"]
    #    mus_fit = npz_data["mus_fit"]
    g = npz_data["g"]

    return {
        'sample_name': name,
        'thickness': thickness,
        'convergence_code': convergence_code,
        'successes': successes,
        'wavelength': wavelength,
        'mua_mean': mua_mean,
        'mua_std': mua_std,
        'mus_mean': mus_mean,
        'mus_std': mus_std,
        #      'mus_fit': mus_fit,
        'g': g,
        'M_R': M_R,
        'M_T': M_T
    }

def get_mcml_sample_data(sample_path: str) -> dict:
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"The file {sample_path} does not exist.")
    folder, name = os.path.split(sample_path)  # e.g. Data/DIS_processed/MCML_data and T.1.1.a
    name = name[:-2]  # e.g. T.1.1
    measurement_path = os.path.join(folder, name + '.a', '1', '')
    mco_files = glob.glob(measurement_path + '*.mco')

    wavelength = []

    M_R = []  # total reflectance
    M_T = []  # total transmittance
    mua_mean = []
    mus_mean = []
    thickness = []
    convergence_code = []
    successes = []
    g = []
    num_measurements = 0
    for file in mco_files:
        parameters = linecache.getline(file, 21).split()
        mua_mean.append(float(parameters[1]))
        mus_mean.append(float(parameters[2]))
        g.append(float(parameters[3]))
        thickness.append(float(parameters[4]) * 10)

        M_RS = float(linecache.getline(file, 25, module_globals=None).split()[0])
        M_RD = float(linecache.getline(file, 26, module_globals=None).split()[0])
        M_R.append(M_RS + M_RD)

        M_T.append(float(linecache.getline(file, 28, module_globals=None).split()[0]))
        convergence_code.append('MCML')
        successes.append(True)
        #  M_A.append(float(linecache.getline(file, 27, module_globals=None).split()[0]))
        wavelength.append(-1)
        num_measurements += 1
    #  if thickness == []:
    #    print('empty output') # a weird bug. MCML sometimes does not produce output due to segfault
    wavelength = np.asarray(wavelength)
    M_R = np.asarray(M_R)
    M_T = np.asarray(M_T)
    mua_mean = np.asarray(mua_mean)
    mus_mean = np.asarray(mus_mean)
    convergence_code = np.asarray(convergence_code)
    successes = np.asarray(successes)
    thickness = np.array(thickness).mean()

    return {
        'sample_name': name,
        'thickness': thickness,
        'convergence_code': convergence_code,
        'successes': successes,
        'wavelength': wavelength,
        'mua_mean': mua_mean,
        'mua_std': 0 * mua_mean,
        'mus_mean': mus_mean,
        'mus_std': 0 * mus_mean,
        #      'mus_fit': mus_fit,
        'g': g,
        'M_R': M_R,
        'M_T': M_T
    }

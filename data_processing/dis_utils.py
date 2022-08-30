import linecache
import os
import pandas as pd
import numpy as np


# read the measurements information from .txt file including the unsuccessful ones
def read_dis_measurement_data(path: str, assumed_anisotropy=0.9):
    data_file = path + "1.txt"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Did not find file in {data_file}. "
                                f"Please name the result file '1.txt' as per convention.")

    dataframe = pd.read_csv(data_file, sep="\t", skiprows=44, header=None)
    result_dict = dict()
    result_dict["wavelength"] = np.asarray(dataframe[0].values)
    result_dict["reflectance"] = np.asarray(dataframe[1].values)
    result_dict["transmittance"] = np.asarray(dataframe[3].values)
    result_dict["absorption"] = np.asarray(dataframe[5].values) * 10  # absorption in 1/cm
    result_dict["scattering"] = np.asarray(dataframe[6].values) * 10 / (1 - assumed_anisotropy)  # scattering in 1/cm
    result_dict["success"] = np.asarray(dataframe[8].values) == "# * "
    result_dict["convergence_code"] = np.asarray(dataframe[8].str[-2])
    result_dict["thickness"] = float(linecache.getline(data_file, 4, module_globals=None).split()[-2])
    return result_dict


def read_spectrum_txt_file(file_path: str) -> tuple:  # no idea what this function is for
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"You need to supply the path to a file, not {file_path}")

    dataframe = pd.read_csv(file_path, sep="\t", header=None)
    return np.asarray(dataframe[0].values), np.asarray(dataframe[1].values)


def read_rxt_file(file_path: str) -> tuple:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"You need to supply the path to a file, not {file_path}")

    dataframe = pd.read_csv(file_path, sep=" ", skiprows=1, header=None)
    return np.asarray(dataframe[0].values), np.asarray(dataframe[1].values), np.asarray(dataframe[2].values)

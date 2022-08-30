import os
import torch
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from data_processing.utils import get_sample_data, get_mcml_sample_data

SPECTRAL_LABELS = [
    'convergence_code',
    'successes',
    'wavelength',
    'mua_mean',
    'mua_std',
    'mus_mean',
    'mus_std',
    #   'mus_fit',
    'g',
    'M_R',
    'M_T']
SAMPLE_LABELS = ['sample_name', 'thickness']
LOG_FEATURES = ['log_mua_mean', 'log_mus_mean', 'log_M_R', 'log_M_T']
NORMALISED_FEATURES = LOG_FEATURES + ["mua_mean", "mus_mean", "thickness", "M_R", "M_T"]

ALL_LABELS = SPECTRAL_LABELS + SAMPLE_LABELS


class DIS_dataset(Dataset):

    # we initialise the dataset by saving the information about every measurement in the data folder to the "data" dictionary
    # sample/phantom is a physical object with certain optical parameters
    # spectrum of each sample is measured several times so that the light passes through several locations (e.g near the edges and through the centre)
    # there are usually 5 different measurements for one sample
    # each measurement produces a spectrum of values depending on the wavelength (usually about 300 data points per measurement from 600 to 900 nm)
    def __init__(self, base_path=os.path.join("Data", "DIS_processed"), data_folder="MCML_even_mua_0_4_mus_no_log",
                 mcml_flag=False):
        self.PROCESSED = os.path.join(base_path, data_folder)
        samples_processed = glob.glob(os.path.join(self.PROCESSED, 'T.*.*.a'))
        samples_processed.sort()

        # The data is stored as individual points with all the relevant info.
        # The information about which sample of measurement it comes from is normally ignored. The only exception is the 'sample_name' field
        self.data = {}
        for label in ALL_LABELS:  # we create a dictionary with labels. Each of them corresponds to an array of values.
            self.data[label] = []
        for sample in samples_processed:  # append data of each sample to the dataset
            try:
                sample_data = get_mcml_sample_data(sample) if mcml_flag else get_sample_data(sample)
            except FileNotFoundError:
                # TODO proper error message to the log
                print("For sample ", sample, " file was not found. Check DIS_dataset class for more info")
                continue

            # if a value is a single int for the sample rather than
            # a spectrum, we copy it so that all the measurements of the same sample have the same value
            for key in SPECTRAL_LABELS:
                self.data[key].append(sample_data[key])
            for key in SAMPLE_LABELS:
                s = sample_data[key]
                strings = []
                for i in range(len(self.data['wavelength'][-1])):
                    strings.append(s)
                self.data[key].append(strings)

        for key in ALL_LABELS:  # unite all the separate arrays for each measurement in a single set
            self.data[key] = np.concatenate(self.data[key]).reshape(-1)
        # record the indices of all successful measurements. This is relevant for wavelength and fit
        self.successful_indices, = np.where(self.data["successes"])
        self.data["M_R_All"] = self.data[
            "M_R"].copy()  # this includes all the values including the ones for which IAD did not converge
        self.data["M_T_All"] = self.data["M_T"].copy()
        self.data["M_R"] = self.data["M_R"][self.successful_indices]
        self.data["M_T"] = self.data["M_T"][self.successful_indices]

        for key in LOG_FEATURES:
            self.data[key] = np.log(self.data[key[4:]] + 1e-5)

        # in order to do normalisation, calculate mean and std
        self.mean = {}
        self.std = {}
        for key in NORMALISED_FEATURES:
            self.mean[key] = self.data[key].mean()
            self.std[key] = self.data[key].std()

    def __len__(self):
        return len(self.data["mua_mean"])

    def __getitem__(self, idx):
        features = ["log_mua_mean", "log_mus_mean"]
        labels = ["M_R", "M_T"]
        x = []
        y = []
        for key in features:
            feature = (self.data[key][idx] - self.mean[key]) / self.std[key]
            x.append(feature)
        for key in labels:
            label = (self.data[key][idx] - self.mean[key]) / self.std[key]
            y.append(label)
        x.append(self.data["thickness"][idx])
        y.append(self.data["thickness"][idx])
        return torch.Tensor(x), \
               torch.Tensor(y)

    def normalize(self, array, key):
        return (array - self.mean[key]) / self.std[key]

    def denormalize(self, array, key):
        return self.mean[key] + self.std[key] * array

    def get_scipy_data(self, include_thickness=True, scale_thickness=1, train_fraction=0.8):
        if include_thickness:
            data = np.concatenate([self.data["mua_mean"].reshape(-1, 1),
                                   self.data["mus_mean"].reshape(-1, 1),
                                   self.data["thickness"][self.successful_indices].reshape(-1, 1) / scale_thickness,
                                   # the scaling factor for KNN is 100 (magic number)
                                   self.data["M_R"].reshape(-1, 1),
                                   self.data["M_T"].reshape(-1, 1),
                                   self.data["thickness"][self.successful_indices].reshape(-1, 1) / scale_thickness], axis=1)
        else:
            data = np.concatenate([self.data["mua_mean"].reshape(-1, 1),
                                   self.data["mus_mean"].reshape(-1, 1),
                                   self.data["M_R"].reshape(-1, 1),
                                   self.data["M_T"].reshape(-1, 1)], axis=1)
        N = len(self)
        train_size = int(train_fraction * N)
        test_size = N - train_size
        train, test = train_test_split(data, train_size=train_size, test_size=test_size)
        input_output_dimensions = 3 if include_thickness else 2
        X_train, X_test = train[:, :input_output_dimensions], test[:, :input_output_dimensions]
        Y_train, Y_test = train[:, input_output_dimensions:], test[:, input_output_dimensions:]
        return X_train, X_test, Y_train, Y_test

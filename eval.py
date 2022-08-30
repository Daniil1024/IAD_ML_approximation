import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import config as c
from models.FCNN import FCNN
from models.MyINN import MyINN
from models.MySmallINN import MySmallINN
from data_processing.DIS_dataset import DIS_dataset

dataset = DIS_dataset(data_folder='MCML_property_space_rectangle', mcml_flag=True)

N = len(dataset)
train_size = int(0.8 * N)
test_size = N - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(
    train_data,
    batch_size=c.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=None,
    pin_memory=False)
test_loader = DataLoader(
    test_data,
    batch_size=c.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=None,
    pin_memory=False)
N_DIM = 2

# a simple chain of operations is collected by ReversibleSequential
inn = FCNN(N_DIM)

inn.load_state_dict(torch.load("models/saved/model_fcnn_mcml_frame_inverse"))

plotted_indices = test_data.indices
flag_log = True
#variables = ["mua_mean", "mus_mean", "M_R", "M_T"]
add_log_if = "log_" if flag_log else ""
#("log_" if flag_log else "") + (variables[i])
with torch.no_grad():
    inn.eval()
    x = torch.cat(
        (torch.reshape(torch.Tensor(dataset.normalize(dataset.data[add_log_if+"mua_mean"], add_log_if+"mua_mean")), (-1, 1)),
            torch.reshape(torch.Tensor(dataset.normalize(dataset.data[add_log_if+"mus_mean"], add_log_if+"mus_mean")), (-1, 1))), axis=1)
    y = torch.cat(
        (torch.reshape(torch.Tensor(dataset.normalize(dataset.data["M_R"], "M_R")), (-1, 1)),
            torch.reshape(torch.Tensor(dataset.normalize(dataset.data["M_T"], "M_T")), (-1, 1))), axis=1)
 #   y_tan = torch.tan((y - center)/(smaller_dist)*(0.45*np.pi))/np.pi
    x_pred = inn(y)
    x_pred[:, 0] = dataset.denormalize(x_pred[:, 0], add_log_if+"mua_mean")
    x_pred[:, 1] = dataset.denormalize(x_pred[:, 1], add_log_if+"mus_mean")
    x[:, 0] = dataset.denormalize(x[:, 0], add_log_if+"mua_mean")
    x[:, 1] = dataset.denormalize(x[:, 1], add_log_if+"mus_mean")
    x_pred = np.exp(x_pred)
    x = np.exp(x)
    y[:, 0] = dataset.denormalize(y[:, 0], "M_R")
    y[:, 1] = dataset.denormalize(y[:, 1], "M_T")
    error = torch.abs(torch.div(x_pred - x, x))
    error = torch.clamp(error, 0, 1)
    print(torch.median(error[:, 0]))
    print(torch.median(error[:, 1]))
    blowup_indices = np.where(error[:, 0] > 20)[0]
   # x[:, 0] = torch.clamp(x[:,0], -13, 10000)
    plt.subplot(2, 2, 1)
    plt.scatter(y[plotted_indices, 0], y[plotted_indices, 1], c=error[plotted_indices, 0], alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.hist(error[plotted_indices, 0], bins=30)
    plt.subplot(2, 2, 3)
    plt.scatter(y[plotted_indices, 0], y[plotted_indices, 1], c=error[plotted_indices, 1], alpha=0.3)
    plt.subplot(2, 2, 4)
    plt.hist(error[plotted_indices, 1], bins=30)
    plt.show()

for i, batch in enumerate(test_loader):
    break
    x, y = batch
    batch_size = x.shape[0]
    if batch_size != c.batch_size:
        continue
    z = torch.cat([y, torch.zeros(c.batch_size, 1)], dim=1)
    x_reconstructed, jac = inn(z, rev=True)
    with torch.no_grad():
        mua = np.exp(dataset.mean["log_mua_mean"] + dataset.std["log_mua_mean"]*x_reconstructed[:, 0])
        mua = np.exp(dataset.mean["log_mus_mean"] + dataset.std["log_mus_mean"]*x_reconstructed[:, 1])
        thickness = dataset.mean["thickness"] + dataset.std["thickness"]*x_reconstructed[:, 2]
        mua_uncertainty = torch.abs(torch.div(mua - np.exp(x[:, 0]), np.exp(x[:, 0]))).mean().numpy()
        mus_uncertainty = torch.abs(torch.div(mua - np.exp(x[:, 1]), np.exp(x[:, 1]))).mean().numpy()
        thickness_uncertainty = torch.abs(torch.div(mua - x[:, 2], x[:, 2])).mean().numpy()
        print(
            "mua unc: ", str(mua_uncertainty),
            "mus unc: ", str(mus_uncertainty),
            "thickness unc: ", str(thickness_uncertainty)
        )
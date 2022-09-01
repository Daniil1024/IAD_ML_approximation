import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.data import DataLoader

import losses
from data_processing.DIS_dataset import DIS_dataset

import torch
import matplotlib.pyplot as plt

# FrEIA imports

import config as c
from ML.models.MyBigFCNN import MyBigFCNN

N_DIM = 3  # mua, mus, thickness

dataset = DIS_dataset(data_folder="2_spheres_2.0-4.0mm", mcml_flag=False)

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
# a simple chain of operations is collected by ReversibleSequential
fcnn = MyBigFCNN(N_DIM)

def noise_batch(ndim):
    return 0*torch.randn(c.batch_size, ndim).to(c.device)


optimizer = torch.optim.Adam(fcnn.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.96)
loss_history = []
for epoch in tqdm(range(7)):
    # a very basic training loop
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # sample data from the moons distribution
        y, x = batch
        x = x.to(c.device)
        y = y.to(c.device)
        batch_size = x.shape[0]
        if batch_size != c.batch_size:
            continue
        extra_dim = x.shape[1] - y.shape[1]
        y = torch.concat([y, noise_batch(extra_dim)], dim=1)

        loss = losses.l2_fit(fcnn(x), y)
        loss.backward()
        epoch_loss += loss.detach().numpy()
        optimizer.step()
        scheduler.step()
    loss_history.append(epoch_loss)
with torch.no_grad():
    fcnn.eval()
    for i, batch in enumerate(test_loader):
        y, x = batch
        out_x = fcnn(x)

        z = out_x[:, :c.ndim_z]
        y_pred = out_x[:, -c.ndim_y:]
        M_R = (dataset.mean["log_mua_mean"] + dataset.std["log_mua_mean"]*y[:, 0])
        M_T = (dataset.mean["log_mus_mean"] + dataset.std["log_mus_mean"]*y[:, 1])
        M_R_pred = (dataset.mean["log_mua_mean"] + dataset.std["log_mua_mean"]*y_pred[:, 0])
        M_T_pred = (dataset.mean["log_mus_mean"] + dataset.std["log_mus_mean"]*y_pred[:, 1])
        M_R = np.exp(M_R)
        M_T = np.exp(M_T)
        M_R_pred = np.exp(M_R_pred)
        M_T_pred = np.exp(M_T_pred)
        fractional_uncertainty = torch.abs(torch.div(M_T - M_T_pred, M_T))
        print(torch.median(fractional_uncertainty))

N = 1
loss_history = np.convolve(loss_history, np.ones(N) / N, mode='valid')

plt.plot(loss_history)
plt.show()
# sample from the INN by sampling from a standard normal and transforming
# it in the reverse direction
print('a')
print('b')
torch.save(fcnn.state_dict(), "models/saved/model_big_fcnn_HQ_inverse")


def visualise_sample_distribution():
    sample_number = dataset.data["sample_number"][dataset.successful_indices]
    muas = dataset.data["mua_mean"]
    muss = dataset.data["mus_mean"]

    M_R = dataset.data["M_R"][dataset.successful_indices]
    M_T = dataset.data["M_T"][dataset.successful_indices]
    fig, axes = plt.subplots(1, 2)
    scatter0 = axes[0].scatter(muas, muss, s=1, c=sample_number, cmap="tab10")
    scatter1 = axes[1].scatter(M_R, M_T, s=1, c=sample_number, cmap="tab10")

    axes[0].set_xlabel("mua")
    axes[0].set_ylabel("mus")
    axes[1].set_xlabel("M_R")
    axes[1].set_ylabel("M_T")

    legend1 = axes[0].legend(*scatter0.legend_elements(),
                             loc="upper right", title="Samples")
    axes[0].add_artist(legend1)
    plt.show()

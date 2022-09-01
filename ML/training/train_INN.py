import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.data import DataLoader

from ML.training import losses
from ML.models.MySmallINN import MySmallINN
from data_processing.DIS_dataset import DIS_dataset

import torch
import matplotlib.pyplot as plt

# FrEIA imports

import config as c

N_DIM = 2  # mua, mus, thickness

dataset = DIS_dataset(data_folder="MCML_property_space_rectangle", mcml_flag=True)

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

def loss_forward_mmd(out, y):
    # Shorten output, and remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :c.ndim_z],
                                   out[:, -c.ndim_y:]), dim=1)
    y_short = torch.cat([y[:, :c.ndim_z], y[:, -c.ndim_y:]], dim=1)

    l_forw_fit = c.lambd_fit_forw * losses.l2_fit(out[:, -c.ndim_y:], y[:, -c.ndim_y:])
    l_forw_mmd = c.lambd_mmd_forw * torch.mean(losses.forward_mmd(output_block_grad, y_short))

    return l_forw_fit, l_forw_mmd
def loss_backward_mmd(x, y):
    x_samples = model.forward(y, rev=True)
    MMD = losses.backward_mmd(x, x_samples)
    if c.mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / c.y_uncertainty_sigma ** 2 * losses.l2_dist_matrix(y, y))
    return c.lambd_mmd_back * torch.mean(MMD)
def loss_reconstruction(out_y, y, x):
    cat_inputs = [out_y[:, :c.ndim_z] + c.add_z_noise * noise_batch(c.ndim_z)]
    if c.ndim_pad_zy:
        cat_inputs.append(out_y[:, c.ndim_z:-c.ndim_y])  # + c.add_pad_noise * noise_batch(c.ndim_pad_zy))
    cat_inputs.append(out_y[:, -c.ndim_y:] + c.add_y_noise * noise_batch(c.ndim_y))

    x_reconstructed = model.forward(x_or_z=torch.cat(cat_inputs, 1), rev=True)
    return c.lambd_reconstruct * losses.l2_fit(x_reconstructed, x)
def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)

model = MySmallINN(N_DIM)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)
scheduler = ExponentialLR(optimizer, gamma=0.98)
loss_history = [[] for _ in range(4)]
for epoch in tqdm(range(30)):
    # a very basic training loop
    losses_epoch = [0] * 4
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = batch
        x = x.to(c.device)
        y = y.to(c.device)
        batch_size = x.shape[0]
        if batch_size != c.batch_size:
            continue
        extra_dim = x.shape[1] - y.shape[1]
        y = torch.concat([y, noise_batch(extra_dim)], dim=1)

        batch_losses = [0, 0, 0, 0]
        out_y = model(x)

        if (True):
            if c.train_forward_mmd:
                r = loss_forward_mmd(out_y, y)
                batch_losses[0] = r[0]
                batch_losses[1] = 0*r[1]
            if c.train_backward_mmd:
                batch_losses[2] = 0*loss_backward_mmd(x, y)
            if c.train_reconstruction:
                batch_losses[3] = loss_reconstruction(out_y.data, y, x)
        else:
            batch_losses[0] = losses.l2_fit(out_y, y)
        l_total = sum(batch_losses)
        l_total.backward()
        for i in range(len(batch_losses)):
            if isinstance(batch_losses[i], torch.Tensor):
                batch_losses[i] = batch_losses[i].detach().numpy()
            losses_epoch[i] += batch_losses[i]
        optimizer.step()
        scheduler.step()
    for i in range(len(loss_history)):
        loss_history[i].append(losses_epoch[i])
model.eval()
flag_log = True
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        y, x = batch
        out_y = model(x, rev=True)

        z = out_y[:, :c.ndim_z]
        y_pred = out_y[:, -c.ndim_y:]
        add_log_if = "log_" if flag_log else ""
        M_R = dataset.denormalize(y[:, 0], add_log_if+"mua_mean")
        M_T = dataset.denormalize(y[:, 1], add_log_if+"mus_mean")
        M_R_pred = dataset.denormalize(y_pred[:, 0], add_log_if+"mua_mean")
        M_T_pred = dataset.denormalize(y_pred[:, 1], add_log_if+"mus_mean")
        if flag_log:
            M_R = np.exp(M_R)
            M_T = np.exp(M_T)
            M_R_pred = np.exp(M_R_pred)
            M_T_pred = np.exp(M_T_pred)
        fractional_uncertainty = torch.abs(torch.div(M_R - M_R_pred, M_R))
   #     plt.subplot(1, 2, 1)
   #     plt.hist(fractional_uncertainty[:], bins=30)
   #     plt.subplot(1, 2, 2)
   #     plt.scatter(x[:, 0], x[:, 1], c=fractional_uncertainty[:], alpha=0.3)
   #     plt.show()
        print(torch.median(fractional_uncertainty))

N = 2
for i in range(4):
    loss_history[i] = np.convolve(loss_history[i], np.ones(N) / N, mode='valid')

plt.plot(loss_history[0][1:], label="forw_fit")
plt.plot(loss_history[1][1:], label="forw_mmd")
plt.plot(loss_history[2][1:], label="back_mmd")
plt.plot(loss_history[3][1:], label="reconstruct")
plt.legend()
plt.show()
# sample from the INN by sampling from a standard normal and transforming
# it in the reverse direction
torch.save(model.state_dict(), "models/saved/model_smallinn_frame_MRMT_normalised")
import numpy as np
import torch
from matplotlib import pyplot as plt

from data_processing.DIS_dataset import DIS_dataset
from models.FCNN import FCNN
from models.knn import KNN
from models.random_forest import RandomForest

header_fontsize = 24


def print_error_legend():
    print("----------------- Sorry, but ... errors encountered ---------------")
    print("   *  ==> Success          ")
    print("  0-9 ==> Monte Carlo Iteration")
    print("   R  ==> M_R is too big   ")
    print("   r  ==> M_R is too small")
    print("   T  ==> M_T is too big   ")
    print("   t  ==> M_T is too small")
    print("   U  ==> M_U is too big   ")
    print("   u  ==> M_U is too small")
    print("   !  ==> M_R + M_T > 1    ")
    print("   +  ==> Did not converge\n")


def visualise_sample_distribution(dataset):
    sample_name = dataset.data["sample_name"]
    _, sample_number = np.unique(sample_name, return_inverse=True)

    M_R = dataset.data["M_R_All"]
    M_T = dataset.data["M_T_All"]
    fig, ax = plt.subplots(1, 1)
    scatter1 = ax.scatter(M_R, M_T, s=1, c=sample_number, cmap="tab20")
    ax.set_title("Data distribution in the measurement space", fontsize=header_fontsize)
    ax.set_xlabel("$M_R$")
    ax.set_ylabel("$M_T$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #    print(values)
  #  legend1 = ax.legend(*scatter1.legend_elements(), loc="upper right", title="Samples")
  #  ax.add_artist(legend1)
    plt.show()


def plot_property_space(dataset, xlim=(-12, 4), ylim=(-10, 8), s=2, alpha=0.3):
    plt.subplot(1, 3, 1)
    plt.xlabel("$\log(\mu_a / cm^{-1})$")
    plt.ylabel("$\log(\mu_s / cm^{-1})$")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("$M_R$", fontsize=header_fontsize)
    lower = 0  # you can change these thresholds to draw only points from certain region of the measurement space
    upper = 1
    plotted_indices = np.where((dataset.data["M_R"] >= lower) & (dataset.data["M_R"] <= upper) &
                               (dataset.data["M_T"] >= lower) & (dataset.data["M_T"] <= upper))
    plt.scatter(dataset.data["log_mua_mean"][plotted_indices],
                dataset.data["log_mus_mean"][plotted_indices],
                vmin=0,
                vmax=1,
                s=s,
                c=dataset.data["M_R"][plotted_indices],
                alpha=alpha)
    plt.subplot(1, 3, 2)
    plt.xlabel("$\log(\mu_a / cm^{-1})$")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("$M_T$", fontsize=header_fontsize)
    plt.scatter(dataset.data["log_mua_mean"][plotted_indices],
                dataset.data["log_mus_mean"][plotted_indices],
                vmin=0,
                vmax=1,
                s=s,
                c=dataset.data["M_T"][plotted_indices],
                alpha=alpha)
    plt.subplot(1, 3, 3)
    plt.xlabel("$\log(\mu_a / cm^{-1})$")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("$M_R+M_T$", fontsize=header_fontsize)
    plt.scatter(dataset.data["log_mua_mean"][plotted_indices],
                dataset.data["log_mus_mean"][plotted_indices],
                vmin=0,
                vmax=1,
                s=s,
                c=(dataset.data["M_T"] + dataset.data["M_R"])[plotted_indices],
                alpha=alpha)
    plt.show()


def plot_measurement_space(dataset, s=2, alpha=0.3):
    xlim = (0, 1)
    ylim = (0, 1)
    plt.subplot(1, 2, 1)
    plt.xlabel("$M_R$")
    plt.ylabel("$M_T$")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("$\log(\mu_a / cm^{-1})$", fontsize=header_fontsize)
    plt.scatter(dataset.data["M_R"].reshape(-1, 1),
                dataset.data["M_T"].reshape(-1, 1),
                vmin=0,
                vmax=1,
                s=s,
                c=dataset.data["log_mua_mean"],
                alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.xlabel("$M_R$")
    plt.ylabel("$M_T$")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("$\log(\mu_s / cm^{-1})$", fontsize=header_fontsize)
    plt.scatter(dataset.data["M_R"].reshape(-1, 1),
                dataset.data["M_T"].reshape(-1, 1),
                vmin=0,
                vmax=1,
                s=s,
                c=dataset.data["log_mus_mean"],
                alpha=alpha)
    plt.show()


def plot_convergence_code_in_measurement_space(dataset, title="Success of IAD for 2-sphere case"):
    convergence_code = dataset.data["convergence_code"]
    values, indices = np.unique(convergence_code, return_inverse=True)
    numerical_convergence_code = indices.tolist()
    scatter = plt.scatter(dataset.data["M_R_All"].reshape(-1, 1), dataset.data["M_T_All"].reshape(-1, 1), s=1,
                          c=numerical_convergence_code, label=values, cmap="tab20")
    plt.title(title, fontsize=header_fontsize)
    plt.xlabel("$M_R$")
    plt.ylabel("$M_T$")
    xlim = (0, 1)
    ylim = (0, 1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(handles=scatter.legend_elements()[0], labels=values.tolist(),
               loc="upper right", title="Success status")
    plt.show()


def plot_bijection(dataset, s=3, alpha=0.3):
    xlim = (-7, 4)
    ylim = (-5, 9)
    plt.subplot(1, 2, 1)
    N_tiles = 50
    color = (np.floor(dataset.data["M_R"] * N_tiles) + np.floor(dataset.data["M_T"] * N_tiles)) % 2
    cmap = 'viridis'
    vmin = 0
    vmax = 1
    plt.xlabel("$\log(\mu_a / cm^{-1})$")
    plt.ylabel("$\log(\mu_s / cm^{-1})$")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("Input space", fontsize=header_fontsize)
    lower = 0
    upper = 1
    plotted_indices = np.where((dataset.data["M_R"] >= lower) & (dataset.data["M_R"] <= upper) &
                               (dataset.data["M_T"] >= lower) & (dataset.data["M_T"] <= upper))
    plt.scatter(dataset.data["log_mua_mean"][plotted_indices],
                dataset.data["log_mus_mean"][plotted_indices],
                vmin=vmin,
                vmax=vmax,
                s=s,
                c=color[plotted_indices],
                cmap=cmap,
                alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.xlabel("$M_R$")
    plt.ylabel("$M_T$")
    #   plt.xlim(0, 1)
    #   plt.ylim(0, 1)
    plt.title("Output space", fontsize=header_fontsize)
    plt.scatter(dataset.data["M_R"][plotted_indices],
                dataset.data["M_T"][plotted_indices],
                vmin=vmin,
                vmax=vmax,
                s=s,
                c=color[plotted_indices],
                cmap=cmap,
                alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.show()

# the following 3 functions depend heaviliy on the exact ML models used, so it is much easier to modify them on the go rather than making them very flexible
def plot_NN_bijection(dataset, xlim=(-11, 4), ylim=(-4, 7), s=2, alpha=0.3):
    model = FCNN(2)
    model.load_state_dict(torch.load("models/saved/model_fcnn_mcml_frame_inverse"))
    model.eval()
    prediction = None
    with torch.no_grad():
        y = torch.cat(
            (torch.reshape(torch.Tensor(dataset.normalize(dataset.data["M_R"], "M_R")), (-1, 1)),
             torch.reshape(torch.Tensor(dataset.normalize(dataset.data["M_T"], "M_T")), (-1, 1))), axis=1)
        output = model.forward(y)
        mua = dataset.denormalize(output[:, 0], "log_mua_mean")
        mus = dataset.denormalize(output[:, 1], "log_mus_mean")
        color = np.floor((dataset.data["M_R"] + dataset.data["M_T"]) * 10)
        cmap = 'tab10'
        lower = 0
        upper = 1
        plotted_indices = np.where((dataset.data["M_R"] >= lower) & (dataset.data["M_R"] <= upper) &
                                   (dataset.data["M_T"] >= lower) & (dataset.data["M_T"] <= upper))
        plt.subplot(1, 2, 1)
        plt.xlabel("$\log(\mu_a)$")
        plt.ylabel("$\log(\mu_s)$")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.title("Input space", fontsize=header_fontsize)
        plt.scatter(mua[plotted_indices],
                    mus[plotted_indices],
                    vmin=0,
                    vmax=10,
                    s=s,
                    c=color[plotted_indices],
                    cmap=cmap,
                    alpha=alpha)
        plt.subplot(1, 2, 2)
        plt.xlabel("$M_R$")
        plt.ylabel("$M_T$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Output space", fontsize=header_fontsize)
        plt.scatter(dataset.data["M_R"][plotted_indices],
                    dataset.data["M_T"][plotted_indices],
                    vmin=0,
                    vmax=10,
                    s=s,
                    c=color[plotted_indices],
                    cmap=cmap,
                    alpha=alpha)
        plt.show()


def plot_forest_bijection(dataset, forest):
    xlim = (-11, 4)
    ylim = (-4, 7)
    plt.subplot(1, 2, 1)

    prediction = None
    with torch.no_grad():
        y = np.concatenate(
            [np.reshape(dataset.data["M_R"], (-1, 1)),
             np.reshape(dataset.data["M_T"], (-1, 1))], axis=1)
        model = forest.model

        output = model.predict(y)
        mua = np.log(output[:, 0])
        mus = np.log(output[:, 1])
        color = np.floor((1 + dataset.data["M_R"] - dataset.data["M_T"]) / 2 * 10)
        cmap = 'tab10'
        lower = 0
        upper = 1
        plotted_indices = np.where((dataset.data["M_R"] >= lower) & (dataset.data["M_R"] <= upper) &
                                   (dataset.data["M_T"] >= lower) & (dataset.data["M_T"] <= upper))
        plt.xlabel("$\log(\mu_a)$")
        plt.ylabel("$\log(\mu_s)$")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.title("Input space")
        plt.scatter(mua[plotted_indices],
                    mus[plotted_indices],
                    vmin=0,
                    vmax=10,
                    s=3,
                    c=color[plotted_indices],
                    cmap=cmap,
                    alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.xlabel("$M_R$")
        plt.ylabel("$M_T$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Output space")
        plt.scatter(dataset.data["M_R"][plotted_indices],
                    dataset.data["M_T"][plotted_indices],
                    vmin=0,
                    vmax=10,
                    s=3,
                    c=color[plotted_indices],
                    cmap=cmap,
                    alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.show()

def plot_NN_prediction_line(dataset, s=2, alpha=0.3):
    model = FCNN(2)
    model.load_state_dict(torch.load("models/saved/model_fcnn_mcml_frame_inverse"))
    model.eval()
    prediction = None
    with torch.no_grad():
        y = torch.cat(
            (torch.reshape(torch.Tensor(dataset.normalize(dataset.data["M_R"], "M_R")), (-1, 1)),
             torch.reshape(torch.Tensor(dataset.normalize(dataset.data["M_T"], "M_T")), (-1, 1))), axis=1)
        output = model.forward(y)
        mua = dataset.data["mua_mean"]
        mus = dataset.data["mus_mean"]
        mua_pred = np.exp(dataset.denormalize(output[:, 0], "log_mua_mean"))
        mus_pred = np.exp(dataset.denormalize(output[:, 1], "log_mus_mean"))
        color = np.floor((dataset.data["M_R"] + dataset.data["M_T"]) * 10)
        cmap = 'tab10'
        lower = 0
        upper = 1
        plotted_indices = np.where((dataset.data["M_R"] >= lower) & (dataset.data["M_R"] <= upper) &
                                   (dataset.data["M_T"] >= lower) & (dataset.data["M_T"] <= upper))
        plt.subplot(1, 2, 1)
        plt.xlabel("$\mu_a$ IAD")
        plt.ylabel("$\mu_a$ FCNN")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.scatter(mua, mua_pred, s=1)

        plt.subplot(1, 2, 2)
        plt.xlabel("$\mu_s$ IAD")
        plt.ylabel("$\mu_s$ FCNN")
        plt.xlim(0, 2000)
        plt.ylim(0, 2000)
        plt.scatter(mus, mus_pred, s=1)
        plt.show()

def plot_forest_prediction_line(dataset, forest, include_thickness=True, model_name='RFR'):
    thickness_scale = 1
    X_train, X_test, Y_train, Y_test = dataset.get_scipy_data(include_thickness=include_thickness, scale_thickness=thickness_scale)
    X_pred = forest.model.predict(Y_test)
    mua_pred = X_pred[:, 0]
    mua = X_test[:, 0]
    mus = X_test[:, 1]
    mus_pred = X_pred[:, 1]
    if include_thickness:
        thickness_pred = X_pred[:, 2] * thickness_scale
        thickness = X_test[:, 2] * thickness_scale
    n_plots = 3 if include_thickness else 2
    plt.subplot(1, n_plots, 1)
    plt.xlabel("$\mu_a$ IAD")
    plt.ylabel("$\mu_a$ " + model_name)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.scatter(mua, mua_pred, s=1)

    plt.subplot(1, n_plots, 2)
    plt.xlabel("$\mu_s$ IAD")
    plt.ylabel("$\mu_s$ " + model_name)
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.scatter(mus, mus_pred, s=1)
    if include_thickness:
        plt.subplot(1, 3, 3)
        plt.xlabel("thickness IAD")
        plt.ylabel("thickness "+model_name)
        plt.scatter(thickness, thickness_pred, s=1, alpha=0.1)

    #    plt.xlim(1.5, 4.5)
    #   plt.ylim(1.5, 4.5)
    plt.show()


def compare_MCML_to_IAD():
    MCML_dataset = DIS_dataset(data_folder="MCML_property_space_rectangle", mcml_flag=True)
    IAD_MCML_dataset = DIS_dataset(data_folder="IAD_MCML_data_reverse", mcml_flag=False)
    successful_indices = IAD_MCML_dataset.successful_indices
    plt.subplot(1, 2, 1)
    plt.xlabel("$\mu_a$ initial")
    plt.ylabel("$\mu_a$ recovered")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.scatter(MCML_dataset.data["mua_mean"][successful_indices], IAD_MCML_dataset.data["mua_mean"], s=1)

    plt.subplot(1, 2, 2)
    plt.xlabel("$\mu_s$ initial")
    plt.ylabel("$\mu_s$ recovered")
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.scatter(MCML_dataset.data["mus_mean"][successful_indices], IAD_MCML_dataset.data["mus_mean"], s=1)
    plt.show()


plt.rcParams['font.size'] = '20'
header_fontsize = 28
my_dataset = DIS_dataset(data_folder="0_spheres_3.0mm", mcml_flag=False)
plot_NN_prediction_line(my_dataset)
#forest = KNN()
#forest.load("models/scipy_saved_models/knn_F")
#plot_forest_prediction_line(my_dataset, forest, include_thickness=True, model_name='KNN')
#plot_convergence_code_in_measurement_space(my_dataset, title="Success of IAD in 2-sphere case")

# forest = RandomForest()
# forest.fit(dataset)
# forest.model = pickle.load(open('RF_HQ', 'rb'))
# knn = KNN()
# knn.model = pickle.load(open('knn_HQ', 'rb'))
# compare_MCML_to_IAD()
# pickle.dump(forest.model, open("RF_HQ_very", 'wb'))
# print(len(dataset))
# plot_hist(dataset)
# print(np.mean(dataset.data["log_mus_mean"]))
# plot_hist(dataset)

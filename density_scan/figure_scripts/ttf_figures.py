import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
from matplotlib import matplotlib_fname

import imaging_methods as im
import cosmoplots as cp
from collections import defaultdict

matplotlib_params = plt.rcParams
cp.set_rcparams_dynamo(matplotlib_params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(matplotlib_params)

results = im.ResultManager.from_json("results.json")
manager = im.GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)
shots = [
    shot
    for shot in manager.get_shot_list()
    if manager.get_discharge_by_shot(shot).comment == "L"
]
apd_data = manager.read_shot_data(shots[0], data_folder="../data")

refx_values = [4, 5, 6, 7]
refy_values = [3, 4, 5, 6]
ylims = [0, 1.5]
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Assuming cp.figure_multiple_rows_columns is similar to plt.subplots


def plot_average_value_vs_gf(param, ax, ylabel, ylims=None):
    gf_groups = defaultdict(list)
    for shot in shots:
        gf = manager.get_discharge_by_shot(shot).greenwald_fraction
        gf_key = round(gf, 1)
        # Compute average lr over all refx and refy values for this shot
        lr = 100 * results.get_blob_param(shot, refx_values, refy_values, param)
        gf_groups[gf_key].append((gf, lr))

    # Compute average lr and gf for each gf group and plot
    gf_avg_values = []
    lr_avg_values = []
    for gf_key, group in gf_groups.items():
        gf_values = [item[0] for item in group]
        lr_values = [item[1] for item in group]
        avg_gf = np.mean(gf_values)
        avg_lr = np.nanmean(lr_values)  # Average lr, ignoring NaN
        gf_avg_values.append(avg_gf)
        lr_avg_values.append(avg_lr)

    # Plot lr vs gf on the second subplot (ax[1])
    ax.scatter(gf_avg_values, lr_avg_values, label=r"$F_{GW}$", color="blue")
    ax.set_xlabel(r"$F_{GW}$")
    ax.set_ylabel(ylabel)
    if ylims is not None:
        ax.set_ylim(ylims)


def make_plot_for_param_and_save(param, ylabel, ylims):
    fig, ax = cp.figure_multiple_rows_columns(1, 1, [])
    plot_average_value_vs_gf(param, ax[0], ylabel=ylabel, ylims=ylims)
    file_name = f"figure_scripts/ttf_{param}.pdf"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close(fig)


make_plot_for_param_and_save("lr", r"$\ell_r$(cm)", ylims=[0, 1])
make_plot_for_param_and_save("lz", r"$\ell_z$(cm)", ylims=[0, 1])
make_plot_for_param_and_save("ly_f", r"$\ell_\perp$(cm)", ylims=[0, 1])
make_plot_for_param_and_save("vx_c", r"$v$(cm/s)", ylims=None)
make_plot_for_param_and_save("vx_2dca_tde", r"$v$(cm/s)", ylims=None)
make_plot_for_param_and_save("vx_tde", r"$v$(cm/s)", ylims=None)

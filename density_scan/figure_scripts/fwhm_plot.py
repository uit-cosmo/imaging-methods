import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import phantom as ph
import cosmoplots as cp
from collections import defaultdict

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

results = ph.ResultManager.from_json("results.json")
manager = ph.PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")
shots = [
    shot
    for shot in manager.get_shot_list()
    if manager.get_discharge_by_shot(shot).confinement_mode == "L"
]
apd_data = manager.read_shot_data(shots[0], data_folder="../data")

refy = [3, 4, 5, 6]
ylims = [0, 1.5]
radial_values = apd_data.R.isel(y=3).values
fig, ax = cp.figure_multiple_rows_columns(1, 2)

gf_groups = defaultdict(list)
for shot in shots:
    gf = manager.get_discharge_by_shot(shot).greenwald_fraction
    gf_key = round(gf, 1)
    lr = [100 * results.get_blob_param(shot, refx, refy, "lr") for refx in range(9)]
    gf_groups[gf_key].append((gf, lr))

for gf_key, group in gf_groups.items():
    gf_values = [item[0] for item in group]
    lr_arrays = [item[1] for item in group]
    avg_gf = np.mean(gf_values)
    avg_lr = np.mean(lr_arrays, axis=0)  # Average across shots for each refx
    ax[0].scatter(radial_values, avg_lr, label=r"$F_{GW}=$" f"${avg_gf:.2f}$")


ax[0].legend()
ax[0].set_xlabel(r"$R$")
ax[0].set_ylabel(r"$\ell_R$(cm)")
ax[0].set_ylim(ylims)

for refx in range(9):
    gfs = [manager.get_discharge_by_shot(shot).greenwald_fraction for shot in shots]
    lrs = [100 * results.get_blob_param(shot, refx, refy, "lr") for shot in shots]
    ax[1].scatter(gfs, lrs, label=f"R={radial_values[refx]:.2f}")

ax[1].set_xlabel(r"$F_{GW}$")
ax[1].set_ylabel(r"$\ell_R$(cm)")
ax[1].legend()
ax[1].set_ylim(ylims)

file_name = os.path.join("result_plots", f"lr_plots.pdf")
plt.savefig(file_name, bbox_inches="tight")
plt.show()
plt.close(fig)


fig, ax = cp.figure_multiple_rows_columns(1, 2)

gf_groups = defaultdict(list)
for shot in shots:
    gf = manager.get_discharge_by_shot(shot).greenwald_fraction
    gf_key = round(gf, 1)
    lz = [100 * results.get_blob_param(shot, refx, refy, "lz") for refx in range(9)]
    gf_groups[gf_key].append((gf, lz))

for gf_key, group in gf_groups.items():
    gf_values = [item[0] for item in group]
    lz_arrays = [item[1] for item in group]
    avg_gf = np.mean(gf_values)
    avg_lz = np.mean(lz_arrays, axis=0)  # Average across shots for each refx
    ax[0].scatter(radial_values, avg_lz, label=r"$F_{GW}=$" f"${avg_gf:.2f}$")

ax[0].legend()
ax[0].set_xlabel(r"$R$")
ax[0].set_ylabel(r"$\ell_Z$(cm)")
ax[0].set_ylim(ylims)

for refx in range(9):
    gfs = [manager.get_discharge_by_shot(shot).greenwald_fraction for shot in shots]
    lzs = [100 * results.get_blob_param(shot, refx, refy, "lz") for shot in shots]
    ax[1].scatter(gfs, lzs, label=f"R={radial_values[refx]:.2f}")

ax[1].set_xlabel(r"$F_{GW}$")
ax[1].set_ylabel(r"$\ell_Z$(cm)")
ax[1].legend()
ax[1].set_ylim(ylims)

file_name = os.path.join("result_plots", f"lz_plots.pdf")
plt.savefig(file_name, bbox_inches="tight")
plt.show()
plt.close(fig)

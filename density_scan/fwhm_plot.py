import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import phantom as ph
import cosmoplots as cp
from collections import defaultdict

full_results = [None for _ in range(90)]
manager = ph.PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")
shots = [shot for shot in manager.get_shot_list() if manager.get_discharge_by_shot(shot).confinement_mode == "L"]
apd_data = manager.read_shot_data(shots[0], data_folder="../data")

for refx in range(9):
    for refy in range(10):
        suffix = f"{refx}{refy}"
        results_file_name = os.path.join("results", f"results_{suffix}.json")
        if os.path.exists(results_file_name):
            full_results[refx*10+refy] = ph.ScanResults.from_json(filename=results_file_name)


def get_results_for_pixel(refx, refy)->ph.ScanResults:
    return full_results[refx*10+refy]


def get_lr_for_refx(refx, shot):
    results = get_results_for_pixel(refx, refy)
    if results is None:
        return np.nan
    return results.shots[shot].blob_params.lr

def get_lz_for_refx(refx, shot):
    results = get_results_for_pixel(refx, refy)
    if results is None:
        return np.nan
    return results.shots[shot].blob_params.lz

refy = 6
radial_values = apd_data.R.isel(y=refy).values
fig, ax = plt.subplots(1, 2)

gf_groups = defaultdict(list)
for shot in shots:
    gf = manager.get_discharge_by_shot(shot).greenwald_fraction
    gf_key = round(gf, 1)
    lr = [100 * get_lr_for_refx(refx, shot) for refx in range(9)]
    gf_groups[gf_key].append((gf, lr))

for gf_key, group in gf_groups.items():
    gf_values = [item[0] for item in group]
    lr_arrays = [item[1] for item in group]
    avg_gf = np.mean(gf_values)
    avg_lr = np.mean(lr_arrays, axis=0)  # Average across shots for each refx
    ax[0].plot(radial_values, avg_lr, label=f"gf={avg_gf:.2f}")


ax[0].legend()
ax[0].set_xlabel(r"$R$")
ax[0].set_ylabel(r"$\ell_R/cm$")
ax[0].set_ylim([0, 3])

for refx in [3, 4, 5, 6, 7, 8]:
    gfs = [manager.get_discharge_by_shot(shot).greenwald_fraction for shot in shots]
    lrs = [100*get_lr_for_refx(refx, shot) for shot in shots]
    ax[1].scatter(gfs, lrs, label=f"R={radial_values[refx]:.2f}")

ax[1].set_xlabel(r"gf")
ax[1].set_ylabel(r"$\ell_R/cm$")
ax[1].legend()
ax[1].set_ylim([0, 3])

file_name = os.path.join("result_plots", f"lr_plots.eps")
plt.savefig(file_name, bbox_inches="tight")
plt.show()
plt.close(fig)


fig, ax = plt.subplots(1, 2)

gf_groups = defaultdict(list)
for shot in shots:
    gf = manager.get_discharge_by_shot(shot).greenwald_fraction
    gf_key = round(gf, 1)
    lz = [100 * get_lz_for_refx(refx, shot) for refx in range(9)]
    gf_groups[gf_key].append((gf, lz))

for gf_key, group in gf_groups.items():
    gf_values = [item[0] for item in group]
    lz_arrays = [item[1] for item in group]
    avg_gf = np.mean(gf_values)
    avg_lz = np.mean(lz_arrays, axis=0)  # Average across shots for each refx
    ax[0].plot(radial_values, avg_lz, label=f"gf={avg_gf:.2f}")

ax[0].legend()
ax[0].set_xlabel(r"$R$")
ax[0].set_ylabel(r"$\ell_Z/cm$")
ax[0].set_ylim([0, 3])

for refx in [3, 4, 5, 6, 7, 8]:
    gfs = [manager.get_discharge_by_shot(shot).greenwald_fraction for shot in shots]
    lrs = [100*get_lz_for_refx(refx, shot) for shot in shots]
    ax[1].scatter(gfs, lrs, label=f"R={radial_values[refx]:.2f}")

ax[1].set_xlabel(r"gf")
ax[1].set_ylabel(r"$\ell_Z/cm$")
ax[1].legend()
ax[1].set_ylim([0, 3])

file_name = os.path.join("result_plots", f"lz_plots.eps")
plt.savefig(file_name, bbox_inches="tight")
plt.show()
plt.close(fig)


import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
from matplotlib import matplotlib_fname

import phantom as ph
import cosmoplots as cp
from collections import defaultdict

matplotlib_params = plt.rcParams
cp.set_rcparams_dynamo(matplotlib_params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(matplotlib_params)

results = ph.ResultManager.from_json("results.json")
manager = ph.PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")
shots = [
    shot
    for shot in manager.get_shot_list()
    if manager.get_discharge_by_shot(shot).confinement_mode == "L"
]
apd_data = manager.read_shot_data(shots[0], data_folder="../data")

refx_values = [4, 5, 6, 7]
refy_values = [3, 4, 5, 6]
ylims = [0, 1.5]

fig, ax = plt.subplots(1, 2)
refx, refy = 6, 6

vx_c = [results.get_blob_param(shot, refx, refy, "vx_c") for shot in shots]
vx_tde = [results.get_blob_param(shot, refx, refy, "vx_tde") for shot in shots]

ax[0].scatter(vx_c, vx_tde, color="blue")
ax[0].set_xlabel("contouring")
ax[0].set_ylabel("TDE")

vy_c = [results.get_blob_param(shot, refx, refy, "vy_c") for shot in shots]
vy_tde = [results.get_blob_param(shot, refx, refy, "vy_tde") for shot in shots]
ax[1].scatter(vy_c, vy_tde, color="blue")
ax[1].set_xlabel("contouring")
ax[1].set_ylabel("TDE")

plt.savefig("velocities_03.pdf", bbox_inches="tight")
plt.show()

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

results = im.ResultManager.from_json("density_scan/results.json")
manager = im.GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)
shot = 1160616016

ds = manager.read_shot_data(shot)
R, Z = ds.R.values, ds.Z.values

fig, ax = plt.subplots()
mask = [
    [True, True, False, True, False, False, True, True, True],
    [False, False, False, False, False, False, False, True, True],
    [True, False, False, False, True, False, False, False, True],
    [False, False, False, True, False, False, False, False, False],
    [True, False, False, False, False, False, False, False, True],
    [False, False, False, False, False, False, False, False, True],
    [False, False, False, False, False, False, False, True, False],
    [False, True, False, True, False, False, False, False, True],
    [True, True, False, False, False, False, False, False, False],
    [False, False, True, False, False, False, False, False, False],
]
mask = xr.DataArray(
    mask[::-1],
    dims=["y", "x"],
    coords={
        "y": range(10),  # y from 0 (top) to 9 (bottom)
        "x": range(9),  # x from 0 to 9
    },
)

for x in range(9):
    for y in range(10):
        ne = results.get_blob_param(shot, x, y, "number_events")
        max_ne = np.nanmax(results.get_blob_param_array(shot, "number_events"))
        min_ne = np.nanmin(results.get_blob_param_array(shot, "number_events"))
        s = 0 if np.isnan(ne) else (ne - min_ne) / max_ne
        ax.scatter(
            ds.R.isel(x=x, y=y).item(),
            ds.Z.isel(x=x, y=y).item(),
            s=10 * s,
            color="green",
        )
        if s == 0:
            ax.scatter(
                ds.R.isel(x=x, y=y).item(),
                ds.Z.isel(x=x, y=y).item(),
                marker="x",
                color="black",
            )
        if mask.isel(x=x, y=y):  # mask[9-y][x]:
            ax.scatter(
                ds.R.isel(x=x, y=y).item(),
                ds.Z.isel(x=x, y=y).item(),
                marker="v",
                color="blue",
            )

plt.show()

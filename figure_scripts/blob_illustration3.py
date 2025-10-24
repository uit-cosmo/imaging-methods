import numpy as np

import matplotlib.pyplot as plt
from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1160616027
manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)
ds = manager.read_shot_data(shot, preprocessed=True)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
plt.rcParams.update(params)

xvals = [5, 6, 7]
yvals = [4, 5, 6]

fig, ax = plt.subplots(
    len(yvals),
    len(xvals),
    figsize=(len(xvals) * 2.08, len(yvals) * 2.58),
    gridspec_kw={"hspace": 0, "wspace": 0},
)

for x in range(len(xvals)):
    for y in range(len(yvals)):
        axe = ax[len(yvals) - y - 1][x]
        average = xr.open_dataset(
            f"density_scan/averages/average_ds_{shot}_{xvals[x]}{yvals[y]}.nc"
        )
        plot_event_with_fit(average, ds, xvals[x], yvals[y], axe)


plt.savefig(f"blob_motion_full_{shot}.pdf", bbox_inches="tight")
plt.show()

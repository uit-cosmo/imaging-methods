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

refy = 6

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

fig, ax = plt.subplots(1, 3, sharey=True)
plt.subplots_adjust(wspace=0.1, hspace=0)


def get_average(shot, refx, refy):
    file_name = os.path.join(
        "density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc"
    )
    if not os.path.exists(file_name):
        print(f"File does not exist {file_name}")
        return None
    average_ds = xr.open_dataset(file_name)
    if len(average_ds.data_vars) == 0:
        return None
    refx_ds, refy_ds = average_ds["refx"].item(), average_ds["refy"].item()
    assert refx == refx_ds and refy == refy_ds
    return average_ds


for refx in np.arange(4, 7):
    axe = ax[(refx - 4)]
    average = get_average(shot, refx, refy)
    plot_2dca_zero_lag(ds, average, axe)

ax[0].set_ylabel(r"$Z$")
ax[0].set_xlabel(r"$R$")
ax[1].set_xlabel(r"$R$")
ax[2].set_xlabel(r"$R$")
ax[2].set_xticks([88, 89, 90, 91])


plt.savefig("cond_av_zero_lag_radial_{}_{}_no_interp.pdf".format(refy, shot), bbox_inches="tight")
plt.show()

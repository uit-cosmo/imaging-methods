import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1160629026
manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)
ds = manager.read_shot_data(shot, preprocessed=True)

refy = 5

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
plt.rcParams.update(params)

fig, ax = plt.subplots(2, 4, figsize=(4 * 2.08, 2 * 2.08), gridspec_kw={"hspace": 0.4})


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


for refx in np.arange(1, 9):
    axe = ax[(refx - 1) // 4][(refx - 1) % 4]
    average = get_average(shot, refx, refy)
    plot_2dca_zero_lag(ds, average, axe)


plt.savefig("cond_av_zero_lag_radial_{}_{}.pdf".format(refy, shot), bbox_inches="tight")
plt.show()

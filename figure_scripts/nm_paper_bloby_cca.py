import numpy as np

import matplotlib.pyplot as plt
from imaging_methods import *
import cosmoplots as cp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import fppanalysis as fppa

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
plt.rcParams.update(params)

shot = 1160616011

ds = manager.read_shot_data(shot)

refx, refy = 2, 5

# ds = ds.sel(time=slice(1.31, 1.32))

average = xr.open_dataset(
    "density_scan/averages/average_ds_{}_{}{}.nc".format(shot, refx, refy)
)
# events, average = find_events_and_2dca(ds, refx, refy, 2, window_size=60, check_max=0, single_counting=True)

fig, ax = plt.subplots(
    3, 3, figsize=(7, 7), gridspec_kw={"hspace": 0.5}
)  # cp.figure_multiple_rows_columns(3, 3)

for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        axe = ax[1 - j][i + 1]
        reference = ds.frames.isel(x=refx, y=refy).values
        data = ds.frames.isel(x=refx + i, y=refy + j).values
        tau_ccf, ccf = fppa.corr_fun(data, reference, 5e-7)
        mask = np.abs(tau_ccf) < 1.5e-5
        tau_ccf, ccf = tau_ccf[mask], ccf[mask]
        R, Z = (
            ds.R.isel(x=refx + i, y=refy + j).item(),
            ds.Z.isel(x=refx + i, y=refy + j).item(),
        )
        if len(average.data_vars) != 0:
            axe.plot(
                average.time.values,
                average.cond_av.isel(x=refx + i, y=refy + j).values
                / average.cond_av.max().item(),
                color="blue",
            )
        axe.plot(tau_ccf, ccf, color="red")
        axe.set_title(r"$R={:.2f}\,Z={:.2f}$".format(R, Z))

plt.savefig("cca_nn_near_sol_{}_{}{}.pdf".format(shot, refx, refy), bbox_inches="tight")
plt.show()

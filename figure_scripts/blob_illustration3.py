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
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

# fig, ax = cp.figure_multiple_rows_columns(2, 4, [None for _ in range(10)])
fig, ax = plt.subplots(
    10, 9, figsize=(9 * 2.08, 10 * 2.08), gridspec_kw={"hspace": 0, "wspace": 0}
)

limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
zfine = np.linspace(-8, 1, 100)

rlcfs, zlcfs = calculate_splinted_LCFS(
    ds["efit_time"].values.mean(),
    ds["efit_time"].values,
    ds["rlcfs"].values,
    ds["zlcfs"].values,
)
dead_data = np.zeros(shape=(10, 9))

for refx in range(9):
    for refy in range(10):
        axe = ax[9 - refy][refx]
        average_fn = os.path
        average = xr.open_dataset(
            f"density_scan/averages/average_ds_{shot}_{refx}{refy}.nc"
        )
        if len(average.data_vars) != 0:
            data = average["cond_av"].sel(time=0).values
        else:
            data = dead_data

        im = axe.imshow(
            data,
            origin="lower",
            interpolation="spline16",
        )
        im.set_clim(0, np.max(data))
        im.set_extent(
            (np.min(ds.R.values) - 0.02, ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0])
        )

        # Make extent slightly larger so that it fits the 88 tick mark
        axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
        axe.plot(rlcfs, zlcfs, color="black")
        axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
        axe.set_xlim((np.min(ds.R.values), ds.R[0, -1]))
        axe.set_xticks([])
        axe.set_yticks([])
        # axe.set_title(r"$R_*={:.2f} $cm".format(ds.R[row, refx]))


plt.savefig(f"blob_motion_full_{shot}.pdf", bbox_inches="tight")
plt.show()

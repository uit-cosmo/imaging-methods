import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1120814031
manager = GPIDataAccessor()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

refy = 5

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

# fig, ax = cp.figure_multiple_rows_columns(2, 4, [None for _ in range(10)])
fig, ax = plt.subplots(2, 4, figsize=(4 * 2.08, 2 * 2.08), gridspec_kw={"hspace": 0.4})

limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
zfine = np.linspace(-8, 1, 100)

rlcfs, zlcfs = calculate_splinted_LCFS(
    ds["efit_time"].values.mean(),
    ds["efit_time"].values,
    ds["rlcfs"].values,
    ds["zlcfs"].values,
)


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
    if average is None:
        continue

    data = average["cond_av"].sel(time=0).values
    im = axe.imshow(
        data,
        origin="lower",
        interpolation="spline16",
    )

    # Make extent slightly larger so that it fits the 88 tick mark
    im.set_extent((np.min(ds.R.values) - 0.02, ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
    im.set_clim(0, np.max(data))
    axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
    axe.plot(rlcfs, zlcfs, color="black")
    axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
    axe.set_xlim((np.min(ds.R.values), ds.R[0, -1]))
    axe.set_xticks([88, 89, 90, 91])
    axe.set_title(r"$R_*={:.2f} $cm".format(ds.R[refy, refx]))


plt.savefig("blob_motion_{}_{}.pdf".format(refy, shot), bbox_inches="tight")
plt.show()

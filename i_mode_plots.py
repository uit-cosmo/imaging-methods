import phantom as ph
from velocity_estimation.correlation import corr_fun
from scipy.optimize import minimize
from scipy import signal
import matplotlib.pyplot as plt
import cosmoplots as cp
import numpy as np
import xarray as xr

shot = 1140613026

manager = ph.PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=False)
refx, refy = 6, 5

# events, average = ph.find_events_and_2dca(
#   ds, refx, refy, threshold=2, check_max=0, window_size=30, single_counting=True
# )
# average.to_netcdf("tmp.nc")
average = xr.open_dataset("tmp.nc")

# fig, ax = plt.subplots(2, 5, figsize=(10, 5))
params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

fig, ax = cp.figure_multiple_rows_columns(2, 5, [None for _ in range(10)])

t_indexes = np.floor(np.linspace(0, average["time"].size - 6, num=10))
# limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
# zfine = np.linspace(-8, 1, 100)

# rlcfs, zlcfs = calculate_splinted_LCFS(
#    ds["efit_time"].values.mean(),
#    ds["efit_time"].values,
#    ds["rlcfs"].values,
#    ds["zlcfs"].values,
# )

for i in np.arange(10):
    # axe = ax[i // 5][i % 5]
    axe = ax[i]
    data = (average["cond_av"].isel(time=int(t_indexes[i])).values,)
    im = axe.imshow(
        average["cond_av"].isel(time=int(t_indexes[i])).values,
        origin="lower",
        interpolation="spline16",
    )
    im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
    im.set_clim(0, np.max(data))
    # im.set_clim(0, 3)
    # axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
    # axe.plot(rlcfs, zlcfs, color="black")
    axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
    axe.set_xlim((ds.R[0, 0], ds.R[0, -1]))
    t = average["time"].isel(time=int(t_indexes[i])).item()
    axe.set_title(r"$t={:.2f}\mu $s".format(t * 1e6))

plt.savefig("blob_motion_114.pdf".format(shot), bbox_inches="tight")
plt.show()

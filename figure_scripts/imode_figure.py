import numpy as np

from phantom.utils import *
import matplotlib.pyplot as plt
from phantom.contours import *
from phantom import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1160616027
shot = 1140613026
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

refx, refy = 6, 5
# events, average = find_events_and_2dca(
#   ds, refx, refy, threshold=2, check_max=0, window_size=60, single_counting=True
# )
# average.to_netcdf("tmp.nc")
average = xr.open_dataset("../tmp.nc")

# show_movie(average, "cond_av")

# fig, ax = plt.subplots(2, 5, figsize=(10, 5))
params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

fig, ax = plt.subplots(2, 4, figsize=(4 * 2.08, 2 * 2.08), gridspec_kw={"hspace": 0.4})

t_indexes = np.floor(np.linspace(15, average["time"].size - 19, num=8))
limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
zfine = np.linspace(-8, 1, 100)

rlcfs, zlcfs = calculate_splinted_LCFS(
    ds["efit_time"].values.mean(),
    ds["efit_time"].values,
    ds["rlcfs"].values,
    ds["zlcfs"].values,
)

for i in np.arange(8):
    # axe = ax[i // 5][i % 5]
    axe = ax[i // 4][i % 4]
    data = (average["cond_av"].isel(time=int(t_indexes[i])).values,)
    im = axe.imshow(
        average["cond_av"].isel(time=int(t_indexes[i])).values,
        origin="lower",
        interpolation="spline16",
    )
    im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
    im.set_clim(0, average["cond_av"].max().item())
    im.set_clim(0, 3)
    axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
    axe.plot(rlcfs, zlcfs, color="black")
    axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
    axe.set_xlim((ds.R[0, 0], ds.R[0, -1]))
    t = average["time"].isel(time=int(t_indexes[i])).item()
    axe.set_title(r"$t={:.2f}\,\mu $s".format(t * 1e6))

plt.savefig("blob_motion_{}.pdf".format(shot), bbox_inches="tight")
plt.show()

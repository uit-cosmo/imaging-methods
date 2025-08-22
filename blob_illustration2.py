import numpy as np

from phantom.utils import *
import matplotlib.pyplot as plt
from phantom.contours import *
from phantom import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1160616027
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

row = 6

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

fig, ax = cp.figure_multiple_rows_columns(2, 4, [None for _ in range(10)])

limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
zfine = np.linspace(-8, 1, 100)

rlcfs, zlcfs = calculate_splinted_LCFS(
    ds["efit_time"].values.mean(),
    ds["efit_time"].values,
    ds["rlcfs"].values,
    ds["zlcfs"].values,
)

for refx in np.arange(1, 9):
    axe = ax[refx - 1]
    events, average = find_events_and_2dca(
        ds, refx, row, threshold=2, check_max=0, window_size=60, single_counting=True
    )

    data = average["cond_av"].sel(time=0).values
    im = axe.imshow(
        data,
        origin="lower",
        interpolation="spline16",
    )

    im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
    im.set_clim(0, np.max(data))
    axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
    axe.plot(rlcfs, zlcfs, color="black")
    axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
    axe.set_xlim((ds.R[0, 0], ds.R[0, -1]))
    axe.set_title(r"$R={:.2f} $cm".format(ds.R[row, refx]))


plt.savefig("blob_motion_{}.pdf".format(shot), bbox_inches="tight")
plt.show()

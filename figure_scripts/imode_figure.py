import numpy as np

import matplotlib.pyplot as plt
from imaging_methods import *
import cosmoplots as cp

shot = 1140613026
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)


limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
zfine = np.linspace(-8, 1, 100)

rlcfs, zlcfs = calculate_splinted_LCFS(
    ds["efit_time"].values.mean(),
    ds["efit_time"].values,
    ds["rlcfs"].values,
    ds["zlcfs"].values,
)


def plot_cond_av(refx, refy):
    average = xr.open_dataset(
        os.path.join("density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc")
    )
    if len(average.data_vars) == 0:
        return
    if refx <= 5:
        t_indexes = np.floor(np.linspace(0, average["time"].size - 7, num=8))
    else:
        t_indexes = np.floor(np.linspace(15, average["time"].size - 19, num=8))
    fig, ax = plt.subplots(
        2, 4, figsize=(4 * 2.08, 2 * 2.08), gridspec_kw={"hspace": 0.4}
    )
    max_data = average.cond_av.max().item()
    for i in np.arange(8):
        # axe = ax[i // 5][i % 5]
        axe = ax[i // 4][i % 4]
        data = average["cond_av"].isel(time=int(t_indexes[i])).values
        im = axe.imshow(
            data,
            origin="lower",
            interpolation="spline16",
        )
        im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
        # im.set_clim(0, np.max(data))
        im.set_clim(0, max_data)
        axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
        axe.plot(rlcfs, zlcfs, color="black")
        axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
        axe.set_xlim((ds.R[0, 0], ds.R[0, -1]))
        t = average["time"].isel(time=int(t_indexes[i])).item()
        axe.set_title(r"$t={:.2f}\,\mu $s".format(t * 1e6))

    R, Z = ds.R.isel(x=refx, y=refy).item(), ds.Z.isel(x=refx, y=refy).item()
    fig.suptitle(r"$R_*={:.2f}\,$cm $\,Z_*={:.2f}\,$cm".format(R, Z), fontsize=16)
    plt.savefig("blob_motion_{}_{}{}.pdf".format(shot, refx, refy), bbox_inches="tight")
    plt.show()


for refx in range(9):
    plot_cond_av(refx, 5)

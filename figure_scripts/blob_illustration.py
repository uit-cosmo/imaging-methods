import numpy as np

import matplotlib.pyplot as plt
from imaging_methods import *
import cosmoplots as cp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
plt.rcParams.update(params)


def plot_cond_av(shot, refx, refy, plot_inset_lcfs=False):
    ds = manager.read_shot_data(shot, preprocessed=True)
    limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
    zfine = np.linspace(-8, 1, 100)

    r_min, r_max, z_lcfs = get_lcfs_min_and_max(ds)
    times_lcfs, r_lcfs = get_average_lcfs_rad_vs_time(ds)

    average = xr.open_dataset(
        os.path.join("density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc")
    )
    if len(average.data_vars) == 0:
        return

    contours = get_contour_evolution(average.cond_av, 0.3)
    t_indexes = np.floor(np.linspace(0, average["time"].size - 7, num=8))
    # t_indexes = np.floor(np.linspace(15, average["time"].size - 19, num=8))
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

        c = contours.contours.isel(time=int(t_indexes[i])).data
        axe.plot(c[:, 0], c[:, 1], ls="--", color="black")

        axe.plot(limit_spline(zfine), zfine, color="black", ls="--")
        axe.fill_betweenx(z_lcfs, r_min, r_max, color="grey", alpha=0.5)
        axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
        axe.set_xlim((ds.R[0, 0], ds.R[0, -1]))

        if i == 0 and plot_inset_lcfs:
            inset_ax = inset_axes(axe, width=0.5, height=0.5, loc="lower left")
            inset_ax.plot(times_lcfs, r_lcfs, color="black")
            inset_ax.set_ylabel(r"$<r_{sep}>$", fontsize=4)
            inset_ax.set_xlabel(r"$t$", fontsize=4)

        t = average["time"].isel(time=int(t_indexes[i])).item()
        axe.set_title(r"$t={:.2f}\,\mu $s".format(t * 1e6))

    R, Z = ds.R.isel(x=refx, y=refy).item(), ds.Z.isel(x=refx, y=refy).item()
    fig.suptitle(r"$R_*={:.2f}\,$cm $\,Z_*={:.2f}\,$cm".format(R, Z), fontsize=16)
    plt.savefig("blob_motion_{}_{}{}.pdf".format(shot, refx, refy), bbox_inches="tight")
    plt.show()


# plot_cond_av(1160616009, 6, 5)
movie_2dca_with_contours(1150916025, 6, 5, run_2dca=True)

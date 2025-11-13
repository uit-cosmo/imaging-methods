import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1160616026
manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)
ds = manager.read_shot_data(shot, preprocessed=True)

refx, refy = 6, 5

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
plt.rcParams.update(params)


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


def movie(
    dataset: xr.Dataset,
    contours_ds,
    com_da,
    max_da,
    variable: str = "cond_av",
    interval: int = 100,
    interpolation: str = "spline16",
    lims=None,
    ax=None,
) -> None:
    t_dim = "time"

    dt = get_dt(dataset)
    refx, refy = int(dataset["refx"].item()), int(dataset["refy"].item())

    def get_title(i):
        time = dataset[t_dim][i]
        if dt < 1e-3:
            title = r"t$={:.2f}\,\mu$s".format(time * 1e6)
        else:
            title = r"t$={:.2f}\,$s".format(time)
        return title

    def animate_2d(i: int) -> Any:
        arr = dataset[variable].isel(**{t_dim: i})
        if lims is None:
            vmin, vmax = np.min(arr), np.max(arr)
        else:
            vmin, vmax = lims
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        c = contours_ds.contours.isel(time=i).data
        line[0].set_data(c[:, 0], c[:, 1])
        com_scatter.set_offsets(com_da.values[i, :])
        max_scatter.set_offsets(max_da.values[i, :])

        tx.set_text(get_title(i))

    if ax is None:
        ax = fig.add_subplot(111)

    tx = ax.set_title(get_title(0))
    ax.scatter(
        dataset.R.isel(x=refx, y=refy).item(),
        dataset.Z.isel(x=refx, y=refy).item(),
        color="black",
    )
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    im = ax.imshow(
        dataset[variable].isel(**{t_dim: 0}),
        origin="lower",
        interpolation=interpolation,
    )
    com_scatter = ax.scatter(
        com_da.values[0, 0], com_da.values[0, 1], marker="s", color="green"
    )
    max_scatter = ax.scatter(
        max_da.values[0, 0], max_da.values[0, 1], marker="^", color="orange"
    )
    line = ax.plot([], [], ls="--", color="black")
    fig.colorbar(im, cax=cax)

    im.set_extent(
        (dataset.R[0, 0], dataset.R[0, -1], dataset.Z[0, 0], dataset.Z[-1, 0])
    )

    ani = animation.FuncAnimation(
        fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
    )

    ani.save("maximum_trajectory.gif", writer="ffmpeg", fps=10)
    plt.show()


average = get_average(shot, refx, refy)
# average = preprocess_average_ds(average, threshold=1)

contour_ds = get_contour_evolution(
    average.cond_av, 0.8, max_displacement_threshold=None, com_method="global"
)

velocity_ds = get_contour_velocity(
    contour_ds.center_of_mass,
    window_size=3,
)
v_c, w_c = velocity_ds.isel(time=slice(10, -10)).mean(dim="time", skipna=True).values

v_f, w_f = get_3tde_velocities(average.cond_av, refx, refy)

maximum_trajectory = compute_maximum_trajectory_da(average)

com_da = contour_ds.center_of_mass

taux, tauy = get_delays(average.cond_av, refx, refy)
deltax = average.R.isel(x=1, y=0).item() - average.R.isel(x=0, y=0).item()
deltay = average.Z.isel(x=0, y=1).item() - average.Z.isel(x=0, y=0).item()
v_2dca, w_wdca = ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)
print("v_2dca_3tde = {:.2f}".format(v_2dca))
print("v_2dca_2tde = {:.2f}".format(deltax / taux))
print("v_c = {:.2f}".format(v_c))


fig, ax = plt.subplots(1, 2)

ax[0].plot(com_da.time.values, com_da.values[:, 0], color="blue")
ax[0].plot(maximum_trajectory.time.values, maximum_trajectory.values[:, 0], color="red")
ax[0].set_title(r"v_c = {:.2f}".format(v_c / 100))
for x in range(9):
    ax[0].hlines(
        average.R.isel(x=x, y=refy).item(),
        np.nanmin(com_da.time.values),
        np.nanmax(com_da.time.values),
    )
ax[0].vlines(taux, np.nanmin(com_da.values[:, 0]), np.nanmax(com_da.values[:, 0]))
ax[0].vlines(-taux, np.nanmin(com_da.values[:, 0]), np.nanmax(com_da.values[:, 0]))

ax[1].plot(com_da.time.values, com_da.values[:, 1], color="blue")
ax[1].plot(maximum_trajectory.time.values, maximum_trajectory.values[:, 1], color="red")
ax[1].set_title(r"v_2dca_tde = {:.2f}".format(v_f / 100))
ax[1].vlines(tauy, np.nanmin(com_da.values[:, 1]), np.nanmax(com_da.values[:, 1]))
ax[1].vlines(-tauy, np.nanmin(com_da.values[:, 1]), np.nanmax(com_da.values[:, 1]))

plt.savefig("trajectories.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
max_cond_av = average.cond_av.max().item()

ax.plot(
    average.time.values,
    average.cond_av.isel(x=refx - 1, y=refy) / max_cond_av,
    color="blue",
)
ax.plot(
    average.time.values, average.cond_av.isel(x=refx, y=refy) / max_cond_av, color="red"
)
ax.plot(
    average.time.values,
    average.cond_av.isel(x=refx + 1, y=refy) / max_cond_av,
    color="black",
)
ax.plot(
    average.time.values, average.cond_repr.max(dim=["x", "y"]), color="black", ls="--"
)


plt.savefig("pixel_intensity.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

fig, ax = plt.subplots()

movie(average, contour_ds, com_da, maximum_trajectory, ax=ax)

# get_average(shot, refx, refy))

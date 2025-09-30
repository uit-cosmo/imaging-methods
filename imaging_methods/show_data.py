"""This module provides functions to create and display animations of model output."""

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from typing import Union, Any
from scipy import interpolate

from .cond_av import find_events_and_2dca
from .utils import *


def get_signal(x, y, data):
    return data.isel(x=x, y=y).dropna(dim="time", how="any")["frames"].values


def get_rz(x, y, data):
    return data.R.isel(x=x, y=y).values, data.Z.isel(x=x, y=y).values


def get_dt(data) -> float:
    return float(data.time[1].values - data.time[0].values)


def show_movie(
    dataset: xr.Dataset,
    variable: str = "frames",
    interval: int = 100,
    gif_name: Union[str, None] = None,
    fps: int = 10,
    interpolation: str = "spline16",
    lims=None,
    fig=None,
    ax=None,
    show=True,
    t_dim="time",
) -> None:
    """
    Creates an animation that shows the evolution of a specific variable over time.

    Parameters
    ----------
    dataset : xr.Dataset
        Model data.
    variable : str, optional
        Variable to be animated (default: "n").
    interval : int, optional
        Time interval between frames in milliseconds (default: 100).
    gif_name : str, optional
        If not None, save the animation as a GIF and name it acoridingly.
    fps : int, optional
        Set the frames per second for the saved GIF (default: 10).

    Returns
    -------
    None

    Notes
    -----
    - This function chooses between a 1D and 2D visualizations based on the dimensionality of the dataset.

    """
    has_limiter = "rlimit" in dataset.coords.keys()
    has_lcfs = "rlcfs" in dataset
    if fig is None:
        fig = plt.figure()

    def animate_2d(i: int) -> Any:
        """
        Create the 2D plot for each frame of the animation.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        None

        """
        arr = dataset[variable].isel(**{t_dim: i})
        if lims is None:
            vmin, vmax = np.min(arr), np.max(arr)
        else:
            vmin, vmax = lims
        # vmin, vmax = 0, 1
        im.set_data(arr)
        # im.set_extent((dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1]))
        im.set_clim(vmin, vmax)
        time = dataset[t_dim][i]
        tx.set_text(r"t $=\,{:.2f}\,\mu$s".format(time * 1e6))
        if has_lcfs:
            rlcfs, zlcfs = calculate_splinted_LCFS(
                time.item(),
                dataset["efit_time"].values,
                dataset["rlcfs"].values,
                dataset["zlcfs"].values,
            )
            lcfs[0].set_data(rlcfs, zlcfs)

    if ax is None:
        ax = fig.add_subplot(111)
    tx = ax.set_title("t = 0")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    t_init = dataset[t_dim][0].values
    im = ax.imshow(
        dataset[variable].isel(**{t_dim: 0}),
        origin="lower",
        interpolation=interpolation,
    )
    fig.colorbar(im, cax=cax)

    if has_limiter:
        limit_spline = interpolate.interp1d(
            dataset["zlimit"], dataset["rlimit"], kind="cubic"
        )
        zfine = np.linspace(-8, 1, 100)
        ax.plot(limit_spline(zfine), zfine, color="black", ls="--")

    if has_lcfs:
        rlcfs, zlcfs = calculate_splinted_LCFS(
            dataset["efit_time"].values.mean(),
            dataset["efit_time"].values,
            dataset["rlcfs"].values,
            dataset["zlcfs"].values,
        )
        lcfs = ax.plot(rlcfs, zlcfs, color="black")

    im.set_extent(
        (dataset.R[0, 0], dataset.R[0, -1], dataset.Z[0, 0], dataset.Z[-1, 0])
    )

    ani = animation.FuncAnimation(
        fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
    )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    if show:
        plt.show()


def calculate_splinted_LCFS(
    time_step: float,
    efit_time: np.ndarray,
    rbbbs: np.ndarray,
    zbbbs: np.ndarray,
):
    rbbbs = np.array(rbbbs)
    zbbbs = np.array(zbbbs)

    time_difference = np.absolute(time_step - efit_time)
    time_index = time_difference.argmin()

    closest_rbbbs = rbbbs[:, time_index]
    closest_zbbbs = zbbbs[:, time_index]

    f = interpolate.interp1d(
        closest_zbbbs[closest_rbbbs >= 86],
        closest_rbbbs[closest_rbbbs >= 86],
        kind="cubic",
    )
    z_fine = np.linspace(-8, 1, 100)
    r_fine = f(z_fine)

    return r_fine, z_fine


def show_movie_with_contours(
    dataset: xr.Dataset,
    contours_ds,
    apd_dataset: xr.Dataset = None,
    variable: str = "n",
    interval: int = 100,
    gif_name: Union[str, None] = None,
    fps: int = 10,
    interpolation: str = "spline16",
    lims=None,
    fig=None,
    ax=None,
    show=True,
    show_debug_info=False,
) -> None:
    """
    Creates an animation that shows the evolution of a specific variable over time.

    Parameters
    ----------
    dataset : xr.Dataset
        Model data.
    variable : str, optional
        Variable to be animated (default: "n").
    interval : int, optional
        Time interval between frames in milliseconds (default: 100).
    gif_name : str, optional
        If not None, save the animation as a GIF and name it acoridingly.
    fps : int, optional
        Set the frames per second for the saved GIF (default: 10).

    Returns
    -------
    None

    Notes
    -----
    - This function chooses between a 1D and 2D visualizations based on the dimensionality of the dataset.

    """
    t_dim = "t" if "t" in dataset._coord_names else "time"
    if fig is None:
        fig = plt.figure()

    dt = get_dt(dataset)
    R, Z = dataset.R.values, dataset.Z.values
    refx, refy = int(dataset["refx"].item()), int(dataset["refy"].item())

    def animate_2d(i: int) -> Any:
        """
        Create the 2D plot for each frame of the animation.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        None

        """
        arr = dataset[variable].isel(**{t_dim: i})
        if lims is None:
            vmin, vmax = np.min(arr), np.max(arr)
        else:
            vmin, vmax = lims
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        c = contours_ds.contours.isel(time=i).data
        line[0].set_data(c[:, 0], c[:, 1])

        time = dataset[t_dim][i]
        if show_debug_info:
            l = contours_ds.length.isel(time=i).item()
            convexity_deficienty = contours_ds.convexity_deficiency.isel(time=i).item()
            com = contours_ds.center_of_mass.isel(time=i).values
            size = contours_ds.area.isel(time=i).item()
            max_displacement = contours_ds["max_displacement"]
            tx.set_text(
                f"l = {l:.2f}, cd = {convexity_deficienty:.2f}, com = {com[0]:.2f} {com[1]:.2f}, area = {size:.2f}, Md = {max_displacement:.2f}"
            )
        else:
            tx.set_text(r"t$={:.2f}\,\mu$s".format(time * 1e6))

    if ax is None:
        ax = fig.add_subplot(111)
    tx = ax.set_title(r"t$={:.2f}\,\mu$s".format(dataset[t_dim][0] * 1e6))
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
    line = ax.plot([], [], ls="--", color="black")
    fig.colorbar(im, cax=cax)

    if apd_dataset is not None:
        limit_spline = interpolate.interp1d(
            apd_dataset["zlimit"], apd_dataset["rlimit"], kind="cubic"
        )
        zfine = np.linspace(-8, 1, 100)
        ax.plot(limit_spline(zfine), zfine, color="black", ls="--")

        r_min, r_max, z_lcfs = get_lcfs_min_and_max(apd_dataset)
        ax.fill_betweenx(z_lcfs, r_min, r_max, color="grey", alpha=0.5)

    im.set_extent(
        (dataset.R[0, 0], dataset.R[0, -1], dataset.Z[0, 0], dataset.Z[-1, 0])
    )

    ani = animation.FuncAnimation(
        fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
    )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    if show:
        plt.show()
    else:
        fig.clf()


def movie_2dca_with_contours(shot, refx, refy, run_2dca=False):
    from .contours import get_contour_evolution
    from .discharge import PlasmaDischargeManager

    manager = PlasmaDischargeManager()
    manager.load_from_json("density_scan/plasma_discharges.json")
    ds = manager.read_shot_data(shot, preprocessed=True)
    ds = ds.sel(time=slice(0.85, 0.96))

    if run_2dca:
        events, average = find_events_and_2dca(
            ds, refx, refy, 2, window_size=60, check_max=1, single_counting=True
        )
    else:
        average = xr.open_dataset(
            "density_scan/averages/average_ds_{}_{}{}.nc".format(shot, refx, refy)
        )

    contour_ds = get_contour_evolution(
        average.cond_av,
        0.3,
        max_displacement_threshold=None,
    )
    output_name = "2dca_{}_{}{}.gif".format(shot, refx, refy)
    fig, ax = plt.subplots(figsize=(3, 3))
    show_movie_with_contours(
        average,
        contour_ds,
        apd_dataset=ds,
        variable="cond_av",
        lims=(0, average.cond_av.max().item()),
        fig=fig,
        ax=ax,
        gif_name=output_name,
        interpolation="spline16",
        show=False,
    )
    fig.clf()


def plot_skewness_and_flatness(ds, shot, fig=None, ax=None):
    from scipy.stats import skew, kurtosis

    if fig is None:
        fig, ax = plt.subplots(
            1, 2, figsize=(2 * 3.3, 1 * 3.3), gridspec_kw={"wspace": 0.5}
        )

    nx = ds.sizes["x"]  # Number of x pixels
    ny = ds.sizes["y"]  # Number of y pixels

    # Initialize array to store skewness
    skewness = np.zeros((ny, nx))
    kurt = np.zeros((ny, nx))

    # Compute skewness for each pixel using a for loop
    for x in range(nx):
        for y in range(ny):
            time_series = ds.frames.isel(x=x, y=y).values
            skewness[y, x] = skew(time_series)
            kurt[y, x] = kurtosis(time_series)

    def plot_image(axe, data, label):
        im = axe.imshow(data, origin="lower", cmap="viridis")
        im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
        axe.set_xlabel("R")
        axe.set_ylabel("Z")
        axe.set_title(f"{label} {shot}")
        plt.colorbar(im, ax=axe, label=label)

    plot_image(ax[0], skewness, "Skewness")
    plot_image(ax[1], kurt, "Kurtosis")


def add_limiter_and_lcfs(ds, ax):
    limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
    zfine = np.linspace(-8, 1, 100)
    ax.plot(limit_spline(zfine), zfine, color="black", ls="--")

    r_min, r_max, z_lcfs = get_lcfs_min_and_max(ds)
    ax.fill_betweenx(z_lcfs, r_min, r_max, color="grey", alpha=0.5)

"""This module provides functions to create and display animations of model output."""

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from typing import Union, Any

import imaging_methods
from .cond_av import find_events_and_2dca
from .utils import *
from .plotting import calculate_splinted_LCFS


def get_signal(x, y, data):
    return data.isel(x=x, y=y).dropna(dim="time", how="any")["frames"].values


def get_rz(x, y, data):
    return data.R.isel(x=x, y=y).values, data.Z.isel(x=x, y=y).values


def get_dt(data) -> float:
    return float(data.time[1].values - data.time[0].values)


def movie_dataset(
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
    normalize=False,
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
        fig = plt.figure(figsize=(5, 5))

    max = dataset[variable].max().item()

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
        normalization = max if normalize else 1
        im.set_data(arr / normalization)
        # im.set_extent((dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1]))
        im.set_clim(vmin, vmax)
        time = dataset[t_dim][i]
        tx.set_text(r"$t/\tau_\text{{d}}=\,{:.1f}$".format(time))
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
    # div = make_axes_locatable(ax)
    # cax = div.append_axes("right", "5%", "5%")
    t_init = dataset[t_dim][0].values
    normalization = max if normalize else 1
    im = ax.imshow(
        dataset[variable].isel(**{t_dim: 0}) / normalization,
        origin="lower",
        interpolation=interpolation,
    )

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
    ax.set_xticks([])
    ax.set_yticks([])

    ani = animation.FuncAnimation(
        fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
    )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    if show:
        plt.show()


def show_movie_with_contours(
    dataset: xr.Dataset,
    contours_ds,
    apd_dataset: xr.Dataset = None,
    variable: str = "frames",
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
        fig = plt.figure(figsize=(5, 5))

    dt = get_dt(dataset)
    R, Z = dataset.R.values, dataset.Z.values
    refx, refy = int(dataset["refx"].item()), int(dataset["refy"].item())
    max = dataset[variable].max().item()

    def get_title(i):
        time = dataset[t_dim][i]
        if dt < 1e-3:
            title = r"t$={:.2f}\,\mu$s".format(time * 1e6)
        else:
            title = r"$t/\tau_\text{{d}}=\,{:.1f}$".format(time)
        return title

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
        im.set_data(arr / max)
        im.set_clim(vmin, vmax)
        c = contours_ds.contours.isel(time=i).data
        line[0].set_data(c[:, 0], c[:, 1])

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
            tx.set_text(get_title(i))

    if ax is None:
        ax = fig.add_subplot(111)

    tx = ax.set_title(get_title(0))
    ax.scatter(
        dataset.R.isel(x=refx, y=refy).item(),
        dataset.Z.isel(x=refx, y=refy).item(),
        color="black",
    )
    im = ax.imshow(
        dataset[variable].isel(**{t_dim: 0}) / max,
        origin="lower",
        interpolation=interpolation,
    )
    line = ax.plot([], [], ls="--", color="black")

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

    ax.set_xticks([])
    ax.set_yticks([])

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    if show:
        plt.show()
    else:
        fig.clf()


def movie_2dca_with_contours(shot, refx, refy, run_2dca=False):
    from .contours import get_contour_evolution
    from .discharge import GPIDataAccessor

    manager = GPIDataAccessor(
        "/home/sosno/Git/experimental_database/plasma_discharges.json"
    )
    ds = manager.read_shot_data(shot, preprocessed=True)
    mp = imaging_methods.get_default_apd_method_params()

    if run_2dca:
        mp.two_dca.refx = refx
        mp.two_dca.refy = refy
        events, average = find_events_and_2dca(ds, mp.two_dca)
    else:
        average = xr.open_dataset(
            "density_scan/averages/average_ds_{}_{}{}.nc".format(shot, refx, refy)
        )
    variable = "cross_corr"
    name = "2dca" if variable == "cond_av" else "2dcc"
    if variable == "cross_corr":
        mp.contouring.threshold_factor = 0.5

    contour_ds = get_contour_evolution(
        average[variable],
        mp.contouring.threshold_factor,
        max_displacement_threshold=None,
    )
    output_name = "{}_{}_{}{}.gif".format(name, shot, refx, refy)
    fig, ax = plt.subplots(figsize=(3, 3))
    show_movie_with_contours(
        average,
        contour_ds,
        apd_dataset=ds,
        variable=variable,
        lims=(0, average[variable].max().item()),
        fig=fig,
        ax=ax,
        gif_name=output_name,
        interpolation=None,  # "spline16",
        show=False,
    )
    fig.clf()

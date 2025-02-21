"""This module provides functions to create and display animations of model output."""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
from matplotlib import animation
from typing import Union, Any
import cosmoplots as cp


def get_signal(x, y, data):
    return data.isel(x=x, y=y).dropna(dim="time", how="any")["frames"].values


def get_rz(x, y, data):
    return data.R.isel(x=x, y=y).values, data.Z.isel(x=x, y=y).values


def get_dt(data) -> float:
    times = data["t"] if "t" in data._coord_names else data["time"]
    return float(times[1].values - times[0].values)


def show_movie(
    dataset: xr.Dataset,
    variable: str = "n",
    interval: int = 100,
    gif_name: Union[str, None] = None,
    fps: int = 10,
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
    fig = plt.figure()

    dt = get_dt(dataset)

    def animate_1d(i: int) -> Any:
        """
        Create the 1D plot for each frame of the animation.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        None

        """
        x = dataset.x
        y = dataset[variable].isel(**{t_dim: i})
        line.set_data(x, y)
        plt.title(f"t = {i*dt:.2f}")

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
        vmax = np.max(arr)
        vmin = np.min(arr)
        vmin, vmax = -1, 3
        im.set_data(arr)
        im.set_extent((dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1]))
        im.set_clim(vmin, vmax)
        time = dataset[t_dim][i]
        tx.set_text(f"t = {time:.7f}")

    if dataset.y.size == 1:
        line, tx = _setup_1d_plot(dataset=dataset, variable=variable)
        ani = animation.FuncAnimation(
            fig, animate_1d, frames=dataset[t_dim].values.size, interval=interval
        )
    else:
        im, tx = _setup_2d_plot(fig=fig, cv0=dataset[variable].isel(**{t_dim: 0}))
        ani = animation.FuncAnimation(
            fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
        )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    plt.show()


def _setup_1d_plot(dataset, variable):
    """
    Set up a 1D plot for the animation.

    Parameters
    ----------
    dataset : xr.Dataset
        Model data.
    variable : str
        Variable to be animated.

    Returns
    -------
    line : matplotlib.lines.Line2D
        Line object representing the plot.
    tx : matplotlib.text.Text
        Text object for the plot title.

    """
    ax = plt.axes(xlim=(0, dataset.x[-1]), ylim=(0, dataset[variable].max()))
    tx = ax.set_title(r"$t = 0$")
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"${variable}$")
    return line, tx


def _setup_2d_plot(fig, cv0):
    """
    Set up a 2D plot for the animation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    cv0 : numpy.ndarray
        Initial 2D array for the plot.

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image object representing the plot.
    tx : matplotlib.text.Text
        Text object for the plot title.

    """
    ax = fig.add_subplot(111)
    tx = ax.set_title("t = 0")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    im = ax.imshow(cv0, origin="lower", interpolation="spline16")
    fig.colorbar(im, cax=cax)
    return im, tx


def show_labels(
    dataset: xr.Dataset,
    variable: str = "n",
    interval: int = 100,
    gif_name: Union[str, None] = None,
    fps: int = 10,
) -> None:
    import matplotlib.colors as mcolors

    t_dim = "t" if "t" in dataset._coord_names else "time"
    fig, axes = plt.subplots(1, 2)

    n_blobs = dataset.blob_labels.number_of_blobs
    base_cmap = plt.cm.viridis
    cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, n_blobs + 1)))
    bounds = np.arange(-0.5, n_blobs + 1.5, 1)

    def animate_2d(i: int) -> Any:
        arr = dataset[variable].isel(**{t_dim: i})
        arr_labels = dataset.blob_labels.isel(**{t_dim: i})
        im_original.set_data(arr)
        im_labels.set_data(arr_labels)
        im_original.set_extent(
            (dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1])
        )
        im_labels.set_extent((dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1]))
        # im_original.set_clim(np.min(arr), np.max(arr))
        time = dataset[t_dim][i]
        title_original.set_text(f"t = {time:.7f}")
        title_labels.set_text(f"t = {time:.7f}")

    title_original = axes[0].set_title("t = 0")
    title_labels = axes[1].set_title("t = 0")
    im_original = axes[0].imshow(
        dataset[variable].isel(**{t_dim: 0}), origin="lower", interpolation="spline16"
    )
    im_labels = axes[1].imshow(
        dataset.blob_labels.isel(**{t_dim: 0}),
        origin="lower",
        cmap=cmap,
        norm=mcolors.BoundaryNorm(bounds, cmap.N),
    )
    ani = animation.FuncAnimation(
        fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
    )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    plt.show()


def _setup_2d_plot(fig, cv0):
    """
    Set up a 2D plot for the animation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    cv0 : numpy.ndarray
        Initial 2D array for the plot.

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image object representing the plot.
    tx : matplotlib.text.Text
        Text object for the plot title.

    """
    ax = fig.add_subplot(111)
    tx = ax.set_title("t = 0")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    im = ax.imshow(cv0, origin="lower", interpolation="spline16")
    fig.colorbar(im, cax=cax)
    return im, tx


def plot_velocity_field(ax, ds):
    import velocity_estimation as ve
    from utils import PhantomDataInterface
    import matplotlib as mpl

    eo = ve.EstimationOptions()
    eo.cc_options.interpolate = True
    eo.neighbour_options.min_separation = 1
    eo.neighbour_options.max_separation = 1

    movie_data = ve.estimate_velocity_field(PhantomDataInterface(ds), eo)
    md_ds = xr.Dataset(
        data_vars=dict(
            vx=(["x", "y"], movie_data.get_vx()), vy=(["x", "y"], movie_data.get_vy())
        ),
        coords=dict(x=ds.x.values, y=ds.y.values),
    )

    vx = movie_data.get_vx()
    vy = movie_data.get_vy()
    confidences = movie_data.get_confidences()
    R = movie_data.get_R()
    Z = movie_data.get_Z()

    plt.rcParams["mathtext.fontset"] = "custom"

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    qiv = ax.quiver(
        R,
        Z,
        vx,
        vy,
        confidences,
        scale=210000,  # Scale arrows
        scale_units="xy",
        angles="xy",
        norm=norm,
    )
    ax.set_xlabel("R [cm]")
    ax.set_ylabel("Z [cm]")
    ax.set_ylim(np.min(Z) - 0.5, np.max(Z) + 0.5)
    ax.set_xlim([np.min(R) - 0.5, np.max(R) + 0.5])

import matplotlib.pyplot as plt
from .utils import *


def plot_2dca_zero_lag(ds, average, ax):
    if average is None or len(average.data_vars) == 0:
        return

    data = average["cond_av"].sel(time=0).values
    im = ax.imshow(data, origin="lower", interpolation=None)  # "spline16",

    half_pixel = (
        0.38 / 2
    )  # The pixel size varies from pixel to pixel, but this is an ok approximation
    im.set_extent(
        (
            np.min(average.R.values) - half_pixel,
            average.R[0, -1] + half_pixel,
            average.Z[0, 0] - half_pixel,
            average.Z[-1, 0] + half_pixel,
        )
    )
    im.set_clim(0, np.max(data))
    ax.set_ylim(
        np.min(average.Z.values) - half_pixel, np.max(average.Z.values) + half_pixel
    )
    ax.set_xlim(
        [np.min(average.R.values) - half_pixel, np.max(average.R.values) + half_pixel]
    )

    ax.set_xticks([88, 89, 90, 91])
    refx, refy = average["refx"].item(), average["refy"].item()
    ax.set_title(r"$R_*={:.2f} $cm".format(average.R[refy, refx]))

    if ds is not None:
        plot_lcfs_area(ds, ax)


def plot_contour_at_zero(e, contour_ds, ax, fig_name=None):
    im = ax.imshow(e.sel(time=0), origin="lower", interpolation="spline16")

    c = contour_ds.contours.sel(time=0).data
    ax.plot(c[:, 0], c[:, 1], ls="--", color="black")

    # Set extent so that the middle of each pixel falls at the coordinates of said pixel.
    pixel = e.R[0, 1] - e.R[0, 0]
    ny, nx = e.R.shape
    minR = np.min(e.R.values)
    minZ = np.min(e.Z.values)
    rmin, rmax, zmin, zmax = (
        minR - pixel / 2,
        minR + (nx - 1 / 2) * pixel,
        minZ - pixel / 2,
        minZ + (ny - 1 / 2) * pixel,
    )
    im.set_extent((rmin, rmax, zmin, zmax))
    area = contour_ds.area.sel(time=0).item()
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")

    return area


def plot_lcfs_area(ds, ax):
    limit_spline = interpolate.interp1d(ds["zlimit"], ds["rlimit"], kind="cubic")
    zfine = np.linspace(-8, 1, 100)
    ax.plot(limit_spline(zfine), zfine, color="silver", ls="--")

    r_min, r_max, z_lcfs = get_lcfs_min_and_max(ds)
    ax.fill_betweenx(z_lcfs, r_min, r_max, color="grey", alpha=0.5)


def plot_lcfs_mean(ds, ax):
    rlcfs, zlcfs = calculate_splinted_LCFS(
        ds["efit_time"].values.mean(),
        ds["efit_time"].values,
        ds["rlcfs"].values,
        ds["zlcfs"].values,
    )
    ax.plot(rlcfs, zlcfs, color="black")


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

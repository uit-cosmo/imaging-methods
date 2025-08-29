import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import velocity_estimation as ve
import warnings


def rotated_blob(params, rx, ry, x, y):
    """
    Compute a rotated 2D Gaussian blob centered at (rx, ry).

    .. math::
        blob(\ell_x, \ell_y, \theta; x, y) = \exp \left( -\frac{(x'-rx)^2}{\ell_x^2}-\frac{(y'-ry)^2}{\ell_y^2} \right)

    where

    .. math::
        x' = (x-rx)\cos\theta + (y-ry)\sin\theta
        y' = -(x-rx)\sin\theta + (y-ry)\cos\theta

    Parameters:
        params (array-like): Parameters [lx, ly, theta] (semi-major/minor axes, rotation angle).
        rx, ry (float): Center pixel indices.
        x, y (ndarray): 2D coordinate grids (shape: ny, nx).

    Returns:
        ndarray: Gaussian blob values (shape: ny, nx).
    """
    lx, ly, theta = params
    xt = (x - rx) * np.cos(theta) + (y - ry) * np.sin(theta)
    yt = (y - ry) * np.cos(theta) - (x - rx) * np.sin(theta)
    return np.exp(-((xt / lx) ** 2) - ((yt / ly) ** 2))


def ellipse_parameters(params, rx, ry, alpha):
    """
    Compute points on the ellipse boundary for visualization.

    Parameters:
        params (array-like): Parameters [lx, ly, theta].
        rx, ry (float): Center pixel indices.
        alpha (float): Curve parameter

    Returns:
        tuple: (xvals, yvals) ellipse boundary points.
    """
    lx, ly, theta = params
    xvals = lx * np.cos(alpha) * np.cos(theta) - ly * np.sin(alpha) * np.sin(theta) + rx
    yvals = lx * np.cos(alpha) * np.sin(theta) + ly * np.sin(alpha) * np.cos(theta) + ry
    return xvals, yvals


def fit_ellipse(
    data,
    rx,
    ry,
    size_penalty_factor=0,
    aspect_ratio_penalty_factor=0,
    theta_penalty_factor=0,
):
    """
    Fit an ellipse to imaging data at pixel (rx, ry) using a Gaussian model. The function tries to minimize:

    .. math::
        E(lx, ly, theta) = \sum (blob(\ell_x, \ell_y, \theta; x, y) - data(x, y))^2 + blob(\ell_x, \ell_y, \theta; x, y)^2(P_s + P_\theta \theta^2+P_\epsilon(1-\ell_x/\ell_y)^2)

    where the mathematical expression of blob is given in the docstring of `rotated_blob`.

    Parameters:
        data (xr.Dataset): Dataset with 'frames' (2D, shape: ny, nx), 'R', 'Z' (2D coordinate grids).
        rx, ry (int): Pixel indices for the ellipse center.
        size_penalty_factor (float): Penalty factor for blob size.
        aspect_ratio_penalty_factor (float): Penalty factor for aspect ratio deviation.
        theta_penalty_factor (float): Penalty factor for theta deviation.

    Returns:
        - lx, ly, theta: Fitted semi-major/minor axes and rotation angle.
    """

    def objective_function(params):
        lx, ly, theta = params
        blob = rotated_blob(params, rx, ry, data.R.values, data.Z.values)
        blob_sum = np.sum(blob**2)
        penalty = blob_sum * (size_penalty_factor + theta_penalty_factor * theta**2)
        aspect_ratio_penalty = (
            blob_sum * (1 - lx / ly) ** 2 * aspect_ratio_penalty_factor
        )
        return np.sum((blob - data.values) ** 2) + penalty + aspect_ratio_penalty

    # Initial guesses for lx, ly, and t
    # Rough estimation
    bounds = [
        (0, 5),  # lx: 0 to 5
        (0, 5),  # ly: 0 to 5
        (0, np.pi / 2),  # t: 0 to 2π
    ]

    result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        popsize=15,  # Optional: population size multiplier
        maxiter=1000,  # Optional: maximum number of iterations
    )

    return result.x


def gaussian_convolve(x, times, s=1.0, kernel_size=None):
    # If kernel_size not specified, use 6*sigma to capture most of the Gaussian
    if kernel_size is None:
        kernel_size = int(6 * s)
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

    center = kernel_size // 2
    kernel = np.exp(-((np.arange(-center, center + 1) / s) ** 2))
    kernel = kernel / kernel.sum()

    return times[center:-center], np.convolve(x, kernel, mode="valid")


def find_maximum_interpolate(x, y):
    from scipy.interpolate import InterpolatedUnivariateSpline
    import warnings

    # Taking the derivative and finding the roots only work if the spline degree is at least 4.
    spline = InterpolatedUnivariateSpline(x, y, k=4)
    possible_maxima = spline.derivative().roots()
    possible_maxima = np.append(
        possible_maxima, (x[0], x[-1])
    )  # also check the endpoints of the interval
    values = spline(possible_maxima)

    max_index = np.argmax(values)
    max_time = possible_maxima[max_index]
    if max_time == x[0] or max_time == x[-1]:
        warnings.warn(
            "Maximization on interpolation yielded a maximum in the boundary!"
        )

    return max_time, spline(max_time)


def get_maximum_time(e, refx=None, refy=None):
    """
    Given an event e find the time at which the maximum amplitude is achieved. Data is convolved with a gaussian with
    standard deviation 3.
    :param e:
    :return:
    """
    if refx is None or refy is None:
        refx, refy = int(e["refx"].item()), int(e["refy"].item())
    convolved_times, convolved_data = gaussian_convolve(
        e.isel(x=refx, y=refy), e.time, s=3
    )
    tau, _ = find_maximum_interpolate(convolved_times, convolved_data)
    if tau <= np.min(convolved_times) or tau >= np.max(convolved_times):
        warnings.warn(
            "Time delay found at the window edge, consider running 2DCA with a larger window"
        )
    return tau


def get_maximum_amplitude(e, x, y):
    convolved_times, convolved_data = gaussian_convolve(e.isel(x=x, y=y), e.time, s=3)
    _, amp = find_maximum_interpolate(convolved_times, convolved_data)
    return amp


def get_3tde_velocities(e, refx, refy):
    taux, tauy = get_delays(e, refx, refy)

    deltax = e.R.isel(x=refx + 1, y=refy).item() - e.R.isel(x=refx, y=refy).item()
    deltay = e.Z.isel(x=refx, y=refy + 1).item() - e.Z.isel(x=refx, y=refy).item()
    return ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)


def get_delays(e, refx, refy):
    ref_time = get_maximum_time(e, refx, refy)
    taux_right = get_maximum_time(e, refx + 1, refy) - ref_time
    taux_left = get_maximum_time(e, refx - 1, refy) - ref_time
    tauy_up = get_maximum_time(e, refx, refy + 1) - ref_time
    tauy_down = get_maximum_time(e, refx, refy - 1) - ref_time
    return (taux_right - taux_left) / 2, (tauy_up - tauy_down) / 2


def plot_event_with_fit(
    e,
    refx,
    refy,
    ax,
    fig_name=None,
    size_penalty_factor=5,
    aspect_ratio_penalty_factor=0.1,
    theta_penalty_factor=0.1,
):
    rx, ry = e.R.isel(x=refx, y=refy).item(), e.Z.isel(x=refx, y=refy).item()
    lx, ly, theta = fit_ellipse(
        e.sel(time=0),
        rx,
        ry,
        size_penalty_factor=size_penalty_factor,
        aspect_ratio_penalty_factor=aspect_ratio_penalty_factor,
        theta_penalty_factor=theta_penalty_factor,
    )
    alphas = np.linspace(0, 2 * np.pi, 200)
    elipsx, elipsy = zip(
        *[ellipse_parameters((lx, ly, theta), rx, ry, a) for a in alphas]
    )
    ax.plot(elipsx, elipsy, color="blue", ls="--")

    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")

    return lx, ly, theta


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

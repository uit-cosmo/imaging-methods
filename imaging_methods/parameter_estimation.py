import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import velocity_estimation as ve
import warnings
from imaging_methods import plot_2dca_zero_lag


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
    By convention lx < ly and theta in (0, pi)
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

    bounds = [
        (1e-10, 5),  # lx
        (1e-10, 5),  # ly
        (0, np.pi / 2),  # theta
    ]

    result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        popsize=15,  # Optional: population size multiplier
        maxiter=1000,  # Optional: maximum number of iterations
    )
    lx, ly, theta = result.x

    if lx > ly:
        return ly, lx, theta + np.pi / 2
    return lx, ly, theta


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


def get_maximum_time(e, refx=None, refy=None, gauss_convolve=False):
    """
    Given an event e find the time at which the maximum amplitude is achieved. Data is convolved with a gaussian with
    standard deviation 3.
    :param e:
    :return:
    """
    if refx is None or refy is None:
        refx, refy = int(e["refx"].item()), int(e["refy"].item())

    is_in_boundaries = 0 <= refx < e.R.sizes["x"] and 0 <= refy < e.R.sizes["y"]
    if not is_in_boundaries:
        return None
    if gauss_convolve:
        times, data = gaussian_convolve(e.isel(x=refx, y=refy), e.time, s=3)
    else:
        times, data = e.time, e.isel(x=refx, y=refy)

    tau, _ = find_maximum_interpolate(times, data)
    if tau <= np.min(times) or tau >= np.max(times):
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

    deltax = e.R.isel(x=1, y=0).item() - e.R.isel(x=0, y=0).item()
    deltay = e.Z.isel(x=0, y=1).item() - e.Z.isel(x=0, y=0).item()
    return ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)


def get_delays(e, refx, refy):
    ref_time = get_maximum_time(e, refx, refy)
    maxtime_right = get_maximum_time(e, refx + 1, refy)
    maxtime_left = get_maximum_time(e, refx - 1, refy)
    maxtime_up = get_maximum_time(e, refx, refy + 1)
    maxtime_down = get_maximum_time(e, refx, refy - 1)
    taus_horizontal = [
        (maxtime_right - ref_time) if maxtime_right is not None else None,
        (ref_time - maxtime_left) if maxtime_left is not None else None,
    ]
    taus_horizontal = [t for t in taus_horizontal if t is not None]

    taus_vertical = [
        (maxtime_up - ref_time) if maxtime_up is not None else None,
        (ref_time - maxtime_down) if maxtime_down is not None else None,
    ]
    taus_vertical = [t for t in taus_vertical if t is not None]
    return (
        np.array(taus_horizontal).mean(),
        np.array(taus_vertical).mean(),
    )


def fit_ellipse_to_event(
    e,
    refx,
    refy,
    size_penalty_factor=5,
    aspect_ratio_penalty_factor=0.1,
    theta_penalty_factor=0.1,
):
    """
    Fits an ellipse to the spatial data of an event at a specific reference point (refx, refy) and time slice (time=0).
    The fitting is performed using a fit_ellipse function (not shown), with penalties applied to control the ellipse's size,
    aspect ratio, and orientation.

    Parameters
    ----------
    e : xr.DataArray
        An xarray DataArray containing spatial data with coordinates R (radial) and Z (vertical)
        at specific x, y, and time dimensions.
    refx : int
        Index of the reference point along the x dimension.
    refy : int
        Index of the reference point along the y dimension.
    size_penalty_factor : float, optional
        Penalty factor for ellipse size, passed to the fit_ellipse function. Default is 5.
    aspect_ratio_penalty_factor : float, optional
        Penalty factor for ellipse aspect ratio, passed to the fit_ellipse function. Default is 0.1.
    theta_penalty_factor : float, optional
        Penalty factor for ellipse orientation (theta), passed to the fit_ellipse function. Default is 0.1.

    Returns
    -------
    lx : float
        Semi-major axis length of the fitted ellipse.
    ly : float
        Semi-minor axis length of the fitted ellipse.
    theta : float
        Orientation angle of the ellipse (in radians).
    By convention lx < ly and theta in (0, pi)
    """
    rx, ry = e.R.isel(x=refx, y=refy).item(), e.Z.isel(x=refx, y=refy).item()
    lx, ly, theta = fit_ellipse(
        e.sel(time=0),
        rx,
        ry,
        size_penalty_factor=size_penalty_factor,
        aspect_ratio_penalty_factor=aspect_ratio_penalty_factor,
        theta_penalty_factor=theta_penalty_factor,
    )
    return lx, ly, theta


def plot_event_with_fit(
    e,
    ds,
    refx,
    refy,
    ax,
    fig_name=None,
    size_penalty_factor=5,
    aspect_ratio_penalty_factor=0.1,
    theta_penalty_factor=0.1,
):
    """
    Plots an ellipse fitted to an event's data at a reference point (refx, refy) on a given matplotlib axis (ax).
    The ellipse is computed using the fit_ellipse_to_event function, and the resulting ellipse is drawn as a dashed blue line.
    Optionally, the plot can be saved to a file if a fig_name is provided.

    Parameters
    ----------
    e : xr.DataSet
        An xarray Dataset containing spatial data with coordinates R (radial) and Z (vertical)
        at specific x, y, and time dimensions.
    refx : int
        Index of the reference point along the x dimension.
    refy : int
        Index of the reference point along the y dimension.
    ax : matplotlib.axes.Axes
        The matplotlib axis object on which to plot the ellipse.
    fig_name : str, optional
        If provided, the plot is saved to a file with this name. The file format is inferred from the extension (e.g., .png, .pdf).
        Default is None.
    size_penalty_factor : float, optional
        Penalty factor for ellipse size in the fitting process, passed to fit_ellipse_to_event. Default is 5.
    aspect_ratio_penalty_factor : float, optional
        Penalty factor for ellipse aspect ratio in the fitting process, passed to fit_ellipse_to_event. Default is 0.1.
    theta_penalty_factor : float, optional
        Penalty factor for ellipse orientation (theta) in the fitting process, passed to fit_ellipse_to_event. Default is 0.1.

    Returns
    -------
    lx : float
        Semi-major axis length of the fitted ellipse.
    ly : float
        Semi-minor axis length of the fitted ellipse.
    theta : float
        Orientation angle of the ellipse (in radians).
    By convention lx < ly and theta in (0, pi)
    """
    if e is None or len(e.data_vars) == 0:
        return None, None, None
    plot_2dca_zero_lag(ds=ds, average=e, ax=ax)
    lx, ly, theta = fit_ellipse_to_event(
        e.cond_av,
        refx,
        refy,
        size_penalty_factor,
        aspect_ratio_penalty_factor,
        theta_penalty_factor,
    )
    rx, ry = e.R.isel(x=refx, y=refy).item(), e.Z.isel(x=refx, y=refy).item()
    alphas = np.linspace(0, 2 * np.pi, 200)
    elipsx, elipsy = zip(
        *[ellipse_parameters((lx, ly, theta), rx, ry, a) for a in alphas]
    )
    ax.plot(elipsx, elipsy, color="blue", ls="--")

    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")

    return lx, ly, theta


def estimate_fwhm_sizes(average_ds):
    """
    Estimates the positions of the full-width half-maximum (FWHM) points in both positive and negative directions
    for radial and poloidal slices of a dataset. This is done by taking 1D slices (rows or columns) through the data
    at reference points and using linear interpolation to find where the values drop to half the peak reference value.
    The function handles both sides of the peak separately, returning signed positions (positive for one side,
    negative for the other). The full FWHM can be computed as the difference between positive and negative positions.

    The function assumes the data is centered at a peak value, with values decreasing monotonically away from the center
    until potentially plateauing or increasing (noise). It identifies the monotonic decreasing segment and interpolates within it.

    Parameters
    ----------
    average_ds : xarray.Dataset or dict-like
        The input dataset containing averaged conditional data. Expected structure:
        - 'cond_av': DataArray with dimensions 'x', 'y', 'time', representing the variable of interest (e.g., conditional average).
        - 'R': DataArray with dimensions 'x', 'y', representing radial coordinates.
        - 'Z': DataArray with dimensions 'x', 'y', representing vertical (poloidal) coordinates.
        - 'refx': Scalar value or DataArray (int), the reference index along the 'x' dimension (radial).
        - 'refy': Scalar value or DataArray (int), the reference index along the 'y' dimension (poloidal).

    Returns
    -------
    rp_fwhm : float
        Position of the half-maximum on the positive radial side (≥ 0).
    rn_fwhm : float
        Position of the half-maximum on the negative radial side (≤ 0).
    zp_fwhm : float
        Position of the half-maximum on the positive poloidal side (≥ 0).
    zn_fwhm : float
        Position of the half-maximum on the negative poloidal side (≤ 0).
    """
    refx, refy = average_ds["refx"].item(), average_ds["refy"].item()
    poloidal_var = average_ds.cond_av.isel(x=refx).sel(time=0).values
    r_ref = average_ds.R.isel(x=refx, y=refy).item()
    z_ref = average_ds.Z.isel(x=refx, y=refy).item()
    poloidal_pos = average_ds.Z.isel(x=refx).values - z_ref
    radial_var = average_ds.cond_av.isel(y=refy).sel(time=0).values
    radial_pos = average_ds.R.isel(y=refy).values - r_ref
    ref_val = poloidal_var[refy]

    def get_last_monotounus_idx(x):
        index = 0
        while index + 1 < len(x):
            if x[index + 1] > x[index]:
                break
            index = index + 1
        return index

    def get_fwhm(values, positions):
        idx = get_last_monotounus_idx(values)
        if idx == 0:
            return 0
        return np.interp(ref_val / 2, values[:idx][::-1], positions[:idx][::-1])

    zp_fwhm = get_fwhm(poloidal_var[refy:], poloidal_pos[refy:])
    zn_fwhm = get_fwhm(
        poloidal_var[: (refy + 1)][::-1], poloidal_pos[: (refy + 1)][::-1]
    )
    rp_fwhm = get_fwhm(radial_var[refx:], radial_pos[refx:])
    rn_fwhm = get_fwhm(radial_var[: (refx + 1)][::-1], radial_pos[: (refx + 1)][::-1])
    assert zp_fwhm >= 0
    assert zn_fwhm <= 0
    assert rp_fwhm >= 0
    assert rn_fwhm <= 0

    return rp_fwhm, rn_fwhm, zp_fwhm, zn_fwhm

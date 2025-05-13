import fppanalysis as fppa
import velocity_estimation as ve
import xarray as xr
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def run_norm_ds(ds, radius):
    """Returns running normalized dataset of a given dataset using run_norm from
    fppanalysis function by applying xarray apply_ufunc.
    Input:
        - ds: xarray Dataset
        - radius: radius of the window used in run_norm. Window size is 2*radius+1. ... int
    'run_norm' returns a tuple of time base and the signal. Therefore, apply_ufunc will
    return a tuple of two DataArray (corresponding to time base and the signal).
    To return a format like the original dataset, we create a new dataset of normalized frames and
    corresponding time computed from apply_ufunc.
    Description of apply_ufunc arguments.
        - first the function
        - then arguments in the order expected by 'run_norm'
        - input_core_dimensions: list of lists, where the number of inner sequences must match
        the number of input arrays to the function 'run_norm'. Each inner sequence specifies along which
        dimension to align the corresponding input argument. That means, here we want to normalize
        frames along time, hence 'time'.
        - output_core_dimensions: list of lists, where the number of inner sequences must match
        the number of output arrays to the function 'run_norm'.
        - exclude_dims: dimensions allowed to change size. This must be set for some reason.
        - vectorize must be set to True in order to for run_norm to be applied on all pixels.
    """
    import xarray as xr

    normalization = xr.apply_ufunc(
        fppa.run_norm,
        ds["frames"],
        radius,
        ds["time"],
        input_core_dims=[["time"], [], ["time"]],
        output_core_dims=[["time"], ["time"]],
        exclude_dims=set(("time",)),
        vectorize=True,
    )

    ds_normalized = xr.Dataset(
        data_vars=dict(
            frames=(["y", "x", "time"], normalization[0].data),
        ),
        coords=dict(
            time=normalization[1].data[0, 0, :],
        ),
    )

    return ds_normalized


def interpolate_nans_3d(ds, time_dim="time"):
    """
    Replace NaN values in a 3D xarray dataset with linear interpolation
    based on neighboring values.

    Parameters:
        ds (xarray.Dataset or xarray.DataArray): Input dataset or data array.
        spatial_dims (tuple): Names of the two spatial dimensions (default: ('x', 'y')).
        time_dim (str): Name of the time dimension (default: 'time').

    Returns:
        xarray.DataArray: Dataset with NaNs replaced by interpolated values.
    """
    from scipy.interpolate import griddata

    def interpolate_2d(array, x, y):
        """Interpolate a 2D array with NaNs using griddata."""
        valid_mask = ~np.isnan(array)
        if np.sum(valid_mask) < 4:
            return array  # Not enough points to interpolate reliably

        valid_points = np.array((x[valid_mask], y[valid_mask])).T
        valid_values = array[valid_mask]

        nan_points = np.array((x[~valid_mask], y[~valid_mask])).T

        # griddata interpolates only for grid points inside the convex hull, otherwise leave values to nan.
        # For values outside the convex hull we use "nearest", which is good enough for plotting purposes.
        interpolated_values = griddata(
            valid_points, valid_values, nan_points, method="linear"
        )
        interpolated_values_nearest = griddata(
            valid_points, valid_values, nan_points, method="nearest"
        )
        interpolated_values[np.isnan(interpolated_values)] = (
            interpolated_values_nearest[np.isnan(interpolated_values)]
        )
        array[~valid_mask] = interpolated_values

    x, y = np.meshgrid(ds["x"], ds["y"], indexing="xy")

    # Iterate over the time dimension and interpolate each 2D slice
    for t in ds[time_dim].values:
        slice_data = ds["frames"].sel({time_dim: t}).values
        interpolate_2d(slice_data, x, y)


class PhantomDataInterface(ve.ImagingDataInterface):
    """Implementation of ImagingDataInterface for xarray datasets given by the
    code at https://github.com/sajidah-ahmed/cmod_functions."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def get_shape(self):
        return self.ds.dims["x"], self.ds.dims["y"]

    def get_signal(self, x: int, y: int):
        return self.ds.isel(x=x, y=y)["frames"].values

    def get_dt(self) -> float:
        times = self.ds["time"]
        return float(times[1].values - times[0].values)

    def get_position(self, x: int, y: int):
        return x, y

    def is_pixel_dead(self, x: int, y: int):
        signal = self.get_signal(x, y)
        return len(signal) == 0 or np.isnan(signal[0])


def get_t_start_end(ds):
    times = ds.time.values
    t_start = times[0]
    t_end = times[len(times) - 1]
    return t_start, t_end


def get_dt(ds):
    times = ds["time"]
    return float(times[1].values - times[0].values)


def plot_ccfs_grid(
    ds, ax, refx, refy, rows, cols, delta, ccf=True, plot_tau_M=False, **kwargs
):
    ref_signal = ds["frames"].isel(x=refx, y=refy).values
    tded = ve.TDEDelegator(
        ve.TDEMethod.CC,
        ve.CCOptions(cc_window=delta, minimum_cc_value=0.2, interpolate=True),
        cache=False,
    )

    for row_ix in range(5):
        y = rows[row_ix]
        for col_ix in range(4):
            x = cols[col_ix]
            signal = ds["frames"].isel(x=x, y=y)
            if ccf:
                time, res = fppa.corr_fun(signal, ref_signal, get_dt(ds))
                tau, c, _ = tded.estimate_time_delay(
                    (x, y), (refx, refy), ve.CModImagingDataInterface(ds)
                )
            else:
                Svals, res, s_var, time, peaks, wait = fppa.cond_av(
                    S=signal,
                    T=ds["time"].values,
                    smin=2,
                    Sref=ref_signal,
                    delta=delta * 2,
                )
                tau, c, _ = tded.estimate_time_delay(
                    (x, y), (refx, refy), ve.CModImagingDataInterface(ds)
                )

            window = np.abs(time) < delta
            ax[row_ix, col_ix].plot(time[window], res[window])

            ax[row_ix, col_ix].set_title(
                "R = {:.2f} Z = {:.2f}".format(ds.R.isel(x=x, y=y), ds.Z.isel(x=x, y=y))
            )
            ax[row_ix, col_ix].vlines(0, -10, 10, ls="--")
            ax[row_ix, col_ix].set_ylim(-0.5, 1)
            multiply = 1 if get_dt(ds) > 1e-3 else 1e6
            if tau is not None:
                ax[row_ix, col_ix].text(
                    x=delta / 2, y=0.5, s=r"$\tau = {:.2f}$".format(tau * multiply)
                )
                if plot_tau_M:
                    vx, vy, lx, ly, theta = (
                        kwargs["vx"],
                        kwargs["vy"],
                        kwargs["lx"],
                        kwargs["ly"],
                        kwargs["theta"],
                    )
                    dx, dy = ds.R.isel(x=x, y=y) - ds.R.isel(x=refx, y=refy), ds.Z.isel(
                        x=x, y=y
                    ) - ds.Z.isel(x=refx, y=refy)
                    ax[row_ix, col_ix].text(
                        x=delta / 2,
                        y=0.7,
                        s=r"$\tau_M = {:.2f}$".format(
                            get_taumax(vx, vy, dx, dy, lx, ly, theta) * multiply
                        ),
                    )


def get_ccf_tau(ds):
    s_ref = ds.frames.isel(x=0, y=0).values
    tau, _ = fppa.corr_fun(ds.frames.isel(x=0, y=0).values, s_ref, dt=get_dt(ds))
    return tau


def get_2d_corr(ds, x, y, delta):
    ref_signal = ds.frames.isel(
        x=x, y=y
    ).values  # Select the time series at (refx, refy)
    tau = get_ccf_tau(ds)

    def corr_wrapper(s):
        tau, res = fppa.corr_fun(
            s, ref_signal, dt=5e-7
        )  # Apply correlation function to each time series
        return res

    ds_corr = xr.apply_ufunc(
        corr_wrapper,
        ds,
        input_core_dims=[
            ["time"]
        ],  # Each function call operates on a single time series
        output_core_dims=[["tau"]],  # Output is also a time array
        vectorize=True,
    )
    ds_corr = ds_corr.assign_coords(tau=tau)
    ds_corr = ds_corr.rename({"tau": "time"})
    trajectory_times = tau[np.abs(tau) < delta]
    return ds_corr.sel(time=trajectory_times)


def rotated_blob(params, rx, ry, x, y):
    lx, ly, t = params
    xt = (x - rx) * np.cos(t) + (y - ry) * np.sin(t)
    yt = (y - ry) * np.cos(t) - (x - rx) * np.sin(t)
    return np.exp(-((xt / lx) ** 2) - ((yt / ly) ** 2))


def ellipse_parameters(params, rx, ry, alpha):
    lx, ly, t = params
    xvals = lx * np.cos(alpha) * np.cos(t) - ly * np.sin(alpha) * np.sin(t) + rx
    yvals = lx * np.cos(alpha) * np.sin(t) + ly * np.sin(alpha) * np.cos(t) + ry
    return xvals, yvals


def fit_ellipse(data, rx, ry, size_penalty_factor=0, aspect_ratio_penalty_factor=0):
    """
    Returns lx, ly, theta
    """
    data_mass = np.sum(data.frames.values**2)

    def model(params):
        blob = rotated_blob(params, rx, ry, data.R.values, data.Z.values)
        blob_sum = np.sum(blob**2)
        penalty = blob_sum * size_penalty_factor
        aspect_ratio_penalty = (
            blob_sum * (1 - params[0] / params[1]) ** 2 * aspect_ratio_penalty_factor
        )
        return np.sum((blob - data.frames.values) ** 2) + penalty + aspect_ratio_penalty

    # Initial guesses for lx, ly, and t
    # Rough estimation
    bounds = [
        (0, 5),  # lx: 0 to 5
        (0, 5),  # ly: 0 to 5
        (-np.pi / 4, np.pi / 4),  # t: 0 to 2π
    ]

    result = differential_evolution(
        model,
        bounds,
        seed=42,  # Optional: for reproducibility
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


def get_maximum_time(e):
    """
    Given an event e find the time at which the maximum amplitude is achieved. Data is convolved with a gaussian with
    standard deviation 3.
    :param e:
    :return:
    """
    refx, refy = int(e["refx"].item()), int(e["refy"].item())
    convolved_times, convolved_data = gaussian_convolve(
        e.frames.isel(x=refx, y=refy), e.time, s=3
    )
    tau, _ = find_maximum_interpolate(convolved_times, convolved_data)
    return tau


def get_maximum_amplitude(e, x, y):
    convolved_times, convolved_data = gaussian_convolve(
        e.frames.isel(x=x, y=y), e.time, s=3
    )
    _, amp = find_maximum_interpolate(convolved_times, convolved_data)
    return amp


def get_3tde_velocities(e):
    refx, refy = int(e["refx"].item()), int(e["refy"].item())
    taux, tauy = get_delays(e, refx, refy)

    deltax = e.R.isel(x=refx + 1, y=refy).item() - e.R.isel(x=refx, y=refy).item()
    deltay = e.Z.isel(x=refx, y=refy + 1).item() - e.Z.isel(x=refx, y=refy).item()
    return ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)


def get_delays(e):
    refx, refy = int(e["refx"].item()), int(e["refy"].item())
    ref_time = get_maximum_time(e, refx, refy)
    taux_right = get_maximum_time(e, refx + 1, refy) - ref_time
    taux_left = get_maximum_time(e, refx - 1, refy) - ref_time
    tauy_up = get_maximum_time(e, refx, refy + 1) - ref_time
    tauy_down = get_maximum_time(e, refx, refy - 1) - ref_time
    return (taux_right - taux_left) / 2, (tauy_up - tauy_down) / 2


def plot_event_with_fit(e, ax, fig_name=None):
    refx, refy = int(e["refx"].item()), int(e["refy"].item())
    rx, ry = e.R.isel(x=refx, y=refy).item(), e.Z.isel(x=refx, y=refy).item()
    lx, ly, theta = fit_ellipse(
        e.sel(time=0), rx, ry, size_penalty_factor=5, aspect_ratio_penalty_factor=1
    )
    im = ax.imshow(e.sel(time=0).frames, origin="lower", interpolation="spline16")
    alphas = np.linspace(0, 2 * np.pi, 200)
    elipsx, elipsy = zip(
        *[ellipse_parameters((lx, ly, theta), rx, ry, a) for a in alphas]
    )
    ax.plot(elipsx, elipsy)

    rmin, rmax, zmin, zmax = (
        e.R[0, 0] - 0.05,
        e.R[0, -1] + 0.05,
        e.Z[0, 0] - 0.05,
        e.Z[-1, 0] + 0.05,
    )
    im.set_extent((rmin, rmax, zmin, zmax))
    ax.set_title(r"$\ell_x={:.2f} \ell_y={:.2f} \theta={:.2f}".format(lx, ly, theta))
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")

    return lx, ly, theta


def plot_2d_ccf(ds, x, y, delta, ax):
    corr_data = get_2d_corr(ds, x, y, delta)
    rx, ry = corr_data.R.isel(x=x, y=y).values, corr_data.Z.isel(x=x, y=y).values
    data = corr_data.sel(tau=0).frames.values

    def model(params):
        blob = rotated_blob(params, rx, ry, corr_data.R.values, corr_data.Z.values)
        return np.sum((blob - data) ** 2)

    # Initial guesses for lx, ly, and t
    # Rough estimation
    bounds = [
        (0, 5),  # lx: 0 to 5
        (0, 5),  # ly: 0 to 5
        (-np.pi / 4, np.pi / 4),  # t: 0 to 2π
    ]

    result = differential_evolution(
        model,
        bounds,
        seed=42,  # Optional: for reproducibility
        popsize=15,  # Optional: population size multiplier
        maxiter=1000,  # Optional: maximum number of iterations
    )

    if ax is not None:
        im = ax.imshow(
            corr_data.sel(tau=0).frames, origin="lower", interpolation="spline16"
        )
        ax.scatter(rx, ry, color="black")
        rmin, rmax, zmin, zmax = (
            corr_data.R[0, 0] - 0.05,
            corr_data.R[0, -1] + 0.05,
            corr_data.Z[0, 0] - 0.05,
            corr_data.Z[-1, 0] + 0.05,
        )

        alphas = np.linspace(0, 2 * np.pi, 200)
        elipsx, elipsy = zip(*[ellipse_parameters(result.x, rx, ry, a) for a in alphas])
        ax.plot(elipsx, elipsy)
        im.set_extent((rmin, rmax, zmin, zmax))

    return result.x


def get_taumax(v, w, dx, dy, lx, ly, t):
    lx_fit, ly_fit = lx, ly
    t_fit = t
    a1 = (dx * ly_fit**2 * v + dy * lx_fit**2 * w) * np.cos(t_fit) ** 2
    a2 = (lx_fit**2 - ly_fit**2) * (dy * v + dx * w) * np.cos(t_fit) * np.sin(t_fit)
    a3 = (dx * lx_fit**2 * v + dy * ly_fit**2 * w) * np.sin(t_fit) ** 2
    d1 = (ly_fit**2 * v**2 + lx_fit**2 * w**2) * np.cos(t_fit) ** 2
    d2 = (lx_fit**2 * v**2 + ly_fit**2 * w**2) * np.sin(t_fit) ** 2
    d3 = 2 * (lx_fit**2 - ly_fit**2) * v * w * np.cos(t_fit) * np.sin(t_fit)
    return (a1 - a2 + a3) / (d1 + d2 - d3)


def get_sample_data(shot, window=None, data_folder="data", preprocessed=True):
    """
    Returns APD data from the specified shot processed by running normalization and 2d interpolation to fill dead pixels.
    :param shot: shot
    :param window: Optionally, total duration time to use
    :param data_folder: Data folder
    :param preprocessed: bool, if True loads preprocessed data
    :return: xr.DataSet containing the APD data
    """
    import os

    file_name = os.path.join(data_folder, f"apd_{shot}.nc")
    if preprocessed:
        file_name = os.path.join(data_folder, f"apd_{shot}_preprocessed.nc")
    try:
        if os.path.exists(file_name):
            ds = xr.open_dataset(file_name)
        else:
            print(f"Warning: File {file_name} not found.")
    except Exception as e:
        print(f"Error reading file {file_name}: {str(e)}")

    ds["frames"] = run_norm_ds(ds, 1000)["frames"]

    if window is not None:
        t_start, t_end = get_t_start_end(ds)
        print("Data with times from {} to {}".format(t_start, t_end))

        t_start = (t_start + t_end) / 2 - window / 2
        t_end = t_start + window
        ds = ds.sel(time=slice(t_start, t_end))

    interpolate_nans_3d(ds)
    return ds

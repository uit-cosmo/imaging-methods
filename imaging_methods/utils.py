import fppanalysis as fppa
import velocity_estimation as ve
import xarray as xr
import numpy as np
from scipy import interpolate
from scipy.signal import windows, convolve
from dataclasses import dataclass, field
from .method_parameters import *


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


def get_dr(ds):
    return ds.R.isel(x=1, y=0).item() - ds.R.isel(x=0, y=0).item()


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


def autocorrelation(times, taud, lam):
    """
    Returns the normalized autocorrelation of a shot noise process.
    Input:
        times:  ndarray, float. Time lag.
        taud: float, pulse duration time.
        lam:  float, pulse asymmetry parameter. Related to pulse rise time by tr = l * td and pulse fall time by tf = (1-l) * tf.
    Output:
        ndarray, float. Autocorrelation at time lag tau.
    """
    assert taud > 0
    assert lam >= 0
    assert lam <= 1

    eps = 1e-8

    if np.abs(lam) < eps or np.abs(lam - 1) < eps:
        return np.exp(-np.abs(times) / taud)

    if np.abs(lam - 0.5) < eps:
        return (1 + 2 * np.abs(times) / taud) * np.exp(-2 * np.abs(times) / taud)

    exp1 = (1 - lam) * np.exp(-np.abs(times) / (taud * (1 - lam)))
    exp2 = lam * np.exp(-np.abs(times) / (taud * lam))
    return (exp1 - exp2) / (1 - 2 * lam)


def power_spectral_density(omega, taud, lam):
    """
    Returns the power spectral density of a shot noise process,
    given by
    PSD(omega) = 2.0 * taud / [(1 + (1 - l)^2 omega^2 taud^2) (1 + l^2 omega^2 taud^2)]
    The power spectral density is normalized such that :math:`\int_0^\inf S(\omega) d\omega = 2\pi`, which adds a factor two to the above equation.
    Input:
        omega...: ndarray, float: Angular frequency
        taud......: float, pulse duration time
        lam.......: float, pulse asymmetry parameter.
    Output:
        psd.....: ndarray, float: Power spectral density
    """
    if taud < 0:
        raise ValueError("Taud must be positive")
    if lam <= 0 or lam >= 1:
        raise ValueError("lam must be in (0, 1)")

    if lam in [0, 1]:
        return 4 * taud / (1 + (taud * omega) * (taud * omega))
    elif lam == 0.5:
        return 64 * taud / (4 + (taud * omega) * (taud * omega)) ** 2

    f1 = 1 + ((1 - lam) * taud * omega) * (1.0 - lam) * taud * omega
    f2 = 1 + (lam * taud * omega) * (lam * taud * omega)
    return 4 * taud / (f1 * f2)


def get_lcfs_min_and_max(ds):
    t_start, t_end = ds.time.min().item(), ds.time.max().item()
    z_fine = np.linspace(-8, 1, 100)
    r_min = 100 * np.ones(len(z_fine))
    r_max = 0 * np.ones(len(z_fine))

    for time_index in range(ds["efit_time"].size):
        time = ds["efit_time"][time_index].item()
        if time < t_start or time > t_end:
            continue
        r_values = ds["rlcfs"].isel(efit_time=time_index).values
        z_values = ds["zlcfs"].isel(efit_time=time_index).values

        f = interpolate.interp1d(
            z_values[r_values >= 80],
            r_values[r_values >= 80],
            kind="cubic",
        )
        r_fine = f(z_fine)
        r_min = np.minimum(r_min, r_fine)
        r_max = np.maximum(r_max, r_fine)

    return r_min, r_max, z_fine


def get_average_lcfs_rad_vs_time(ds):
    t_start, t_end = ds.time.min().item(), ds.time.max().item()
    z_min_apd, z_max_apd = ds.Z[0, 0], ds.Z[-1, 0]
    z_fine = np.linspace(z_min_apd, z_max_apd, 100)

    etimes = ds["efit_time"]
    mask = np.logical_and(etimes > t_start, etimes < t_end)
    times_in_range = etimes[mask]
    rvals = []

    for time in times_in_range:
        if time < t_start or time > t_end:
            continue
        r_values = ds["rlcfs"].sel(efit_time=time).values
        z_values = ds["zlcfs"].sel(efit_time=time).values

        f = interpolate.interp1d(
            z_values[r_values >= 80],
            r_values[r_values >= 80],
            kind="cubic",
        )
        r_fine = f(z_fine)
        rvals.append(r_fine.mean())

    return times_in_range, np.array(rvals)


def restrict_to_largest_true_subarray(mask):
    """
    Restrict the True values in the mask to the range of the longest consecutive True subarray.

    Parameters:
        mask (np.ndarray): A boolean array.

    Returns:
        np.ndarray: A new boolean mask with True values only in the range of the longest consecutive True subarray.
    """
    # Convert the boolean array to integers (True -> 1, False -> 0)
    mask_int = mask.astype(int)

    # Find the start and end indices of the longest consecutive True subarray
    diff = np.diff(np.concatenate(([0], mask_int, [0])))  # Add padding to detect edges
    starts = np.where(diff == 1)[0]  # Indices where True starts
    ends = np.where(diff == -1)[0]  # Indices where True ends

    # Calculate lengths of consecutive True segments
    lengths = ends - starts

    if len(lengths) == 0:
        # No True values in the mask
        return np.zeros_like(mask, dtype=bool)

    # Find the range of the longest consecutive True subarray
    max_index = lengths.argmax()
    start_idx = starts[max_index]
    end_idx = ends[max_index] - 1  # End index is inclusive

    # Create a new mask with True values only in the range [start_idx, end_idx]
    restricted_mask = np.zeros_like(mask, dtype=bool)
    restricted_mask[start_idx : end_idx + 1] = True

    return restricted_mask


def smooth_da(
    da: xr.DataArray, pos_filter: PositionFilterParams, return_start_end=False
):
    """
    Smooth a dataarray with a time coordinate with a filter with settings given by pos_filter. The nan values of the
    dataarray are first interpolated. If return_start_end is True, it returns the time coordinate indexes were the
    resulting dataarray is valid.
    :param da: DataArray to be smoothed
    :param pos_filter: Filter settings
    :param return_start_end: If return_start_end is True, it returns the time coordinate indexes were the
    resulting dataarray is valid.
    :return: Smoothed dataarray
    """
    if not isinstance(pos_filter, PositionFilterParams):
        raise ValueError("pos_filter must be a PositionFilterParams")
    if len(da.time) < 2:
        raise ValueError("At least two time points are required")
    window_size = pos_filter.window_size
    window_type = pos_filter.window_type
    # First interpolate nan values
    values_interp = da.interpolate_na(
        dim="time", method="linear", fill_value="extrapolate"
    ).values

    # Define window parameters
    half_window = window_size // 2
    n_times = len(da.time)
    start = half_window
    end = n_times - half_window

    # Check if there are enough points
    if start >= end:
        raise ValueError(
            f"Not enough time points to compute velocity with window_size={window_size}"
        )

    # Generate window using scipy.signal.windows
    if window_type == "gaussian":
        window = windows.gaussian(window_size, std=window_size / 4, sym=True)
    else:
        window = getattr(windows, window_type)(window_size, sym=True)
    window /= window.sum()  # Normalize

    smoothed = convolve(values_interp, window[:, np.newaxis], mode="valid")

    result = xr.DataArray(
        smoothed,
        dims=("time", "coord"),
        coords={"time": da.time[start:end], "coord": ["r", "z"]},
    )
    if return_start_end:
        return result, start, end
    return result


def get_default_synthetic_method_params() -> MethodParameters:
    method_parameters = MethodParameters(
        preprocessing=PreprocessingParams(radius=1000),
        two_dca=TwoDcaParams(
            refx=8, refy=8, threshold=2, window=60, check_max=1, single_counting=True
        ),
        gauss_fit=GaussFitParams(size_penalty=5, aspect_penalty=0.2, tilt_penalty=0.2),
        contouring=ContouringParams(threshold_factor=0.5),
        taud_estimation=TaudEstimationParams(cutoff=1e6, nperseg=2e3),
        position_filter=PositionFilterParams(11, "hann", 2, 0.75),
    )

    return method_parameters


def get_default_apd_method_params() -> MethodParameters:
    method_parameters = MethodParameters(
        preprocessing=PreprocessingParams(radius=1000),
        two_dca=TwoDcaParams(
            refx=8, refy=8, threshold=2, window=60, check_max=1, single_counting=True
        ),
        gauss_fit=GaussFitParams(size_penalty=5, aspect_penalty=0.2, tilt_penalty=0.2),
        contouring=ContouringParams(threshold_factor=0.3),
        taud_estimation=TaudEstimationParams(cutoff=1e6, nperseg=2e3),
        position_filter=PositionFilterParams(11, "hann", 2, 0.75),
    )

    return method_parameters

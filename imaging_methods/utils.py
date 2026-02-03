import velocity_estimation as ve
import xarray as xr
import numpy as np
from scipy.signal import windows, convolve
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


def get_dt(ds):
    times = ds["time"]
    return float(times[1].values - times[0].values)


def get_dr(ds):
    return ds.R.isel(x=1, y=0).item() - ds.R.isel(x=0, y=0).item()


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

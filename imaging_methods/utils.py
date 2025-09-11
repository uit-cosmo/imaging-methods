import fppanalysis as fppa
import velocity_estimation as ve
import xarray as xr
import numpy as np


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


def validate_dataset(ds):
    """
    Check for expected dataset structure
    :param ds: Dataset
    """
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise ValueError("Input 'event' must be an xarray Dataset or DataArray")
    if "frames" not in ds and not isinstance(ds, xr.DataArray):
        raise ValueError("Input must contain 'frames' or be a DataArray")
    if "R" not in ds.coords or "Z" not in ds.coords:
        raise ValueError("Input must include 'R' and 'Z' coordinates")


def validate_dataarray(da, coords=False):
    """
    Check for expected dataarray structure
    :param da: DataArray
    :param coords: bool, if True, checks for coords dimension ('r' and 'z')
    """
    if not isinstance(da, xr.DataArray):
        raise ValueError("Input 'com_da' must be an xarray DataArray")
    if "time" not in da.dims:
        raise ValueError("Input must have time dimensions")
    if "time" not in da.coords:
        raise ValueError("Input must include 'time' coordinates")
    if coords:
        if "coord" not in da.dims or da.shape[-1] != 2:
            raise ValueError("Input must have coord dimension with coord=['r', 'z']")
        if "coord" not in da.coords:
            raise ValueError("Input must include 'coord' coordinates")
        if not np.array_equal(da.coord.values, ["r", "z"]):
            raise ValueError("Input coord must be ['r', 'z']")


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

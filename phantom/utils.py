import fppanalysis as fppa
import velocity_estimation as ve
import xarray as xr
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

from phantom import rotated_blob, ellipse_parameters
from phantom.parameter_estimation import fit_ellipse


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

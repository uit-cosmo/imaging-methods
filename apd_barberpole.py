import numpy as np

from phantom.show_data import *
from phantom.utils import *
import fppanalysis as fpp
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import velocity_estimation as ve

shot = 1140613026

ds = xr.open_dataset("ds_short.nc")
t_start, t_end = get_t_start_end(ds)
t_start = (t_start + t_end) / 2
t_end = t_start + 0.005
# ds = ds.sel(time=slice(t_start, t_end))

times = ds["time"].values
refx, refy = 7, 5


def plot_ccfs():
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 20))
    ref_signal = ds["frames"].isel(x=refx, y=refy).values
    delta = 1e-5

    for row in range(4):
        ax[row].set_title("Col {}".format(5 + row))
        ax[row].vlines(0, -10, 10, ls="--")
        ax[row].set_ylim(-0.5, 1.1)
        for j in [3, 4, 5, 6, 7]:
            signal = ds["frames"].isel(x=refx - 2 + row, y=j)
            time, res = fpp.corr_fun(signal, ref_signal, 5e-7)
            window = np.abs(time) < delta
            # Svals, s_av, s_var, t_av, peaks, wait = fpp.cond_av(S=get_signal(j, refx-2+row), T=times, smin=2, Sref=get_signal(refy, refx), delta=5e-5)
            ax[row].plot(
                time[window],
                res[window],
                label="Z={:.2f}".format(ds.Z.isel(x=refx - 2 + row, y=j).values),
            )

    ax[0].legend(loc=5)

    plt.savefig("ref72_ccf.eps", bbox_inches="tight")
    plt.show()


delta = 1e-5
eo = ve.EstimationOptions(cc_options=ve.CCOptions(cc_window=delta, minimum_cc_value=0.5, interpolate=True))
tded = ve.TDEDelegator(ve.TDEMethod.CC, ve.CCOptions(cc_window=delta, minimum_cc_value=0.2, interpolate=True), cache=False)


def plot_ccfs_grid(ccf=True):
    fig, ax = plt.subplots(5, 4, sharex=True, figsize=(10, 20))
    ref_signal = ds["frames"].isel(x=refx, y=refy).values

    for row in range(5):
        y = 7 - row
        for column in range(4):
            x = column + 5
            signal = ds["frames"].isel(x=x, y=y)
            if ccf:
                time, res = fpp.corr_fun(signal, ref_signal, 5e-7)
                tau, c, _ = tded.estimate_time_delay((x, y), (refx, refy), ve.CModImagingDataInterface(ds))
            else:
                Svals, res, s_var, time, peaks, wait = fpp.cond_av(S=signal, T=ds["time"].values, smin=2, Sref=ref_signal, delta=delta*2)
                tau, c, _ = tded.estimate_time_delay((x, y), (refx, refy), ve.CModImagingDataInterface(ds))

            window = np.abs(time) < delta
            ax[row, column].plot(time[window], res[window])

            ax[row, column].set_title("R = {:.2f} Z = {:.2f}".format(ds.R.isel(x=x, y=y), ds.Z.isel(x=x,y=y)))
            ax[row, column].vlines(0, -10, 10, ls="--")
            ax[row, column].set_ylim(-0.5, 1.1 * np.max(res))
            ax[row, column].text(x=delta/2, y = 0.5, s=r"$\tau = {:.2f}$".format(tau*1e6))

    name = "grid_ccf{}{}.eps".format(refx, refy) if ccf else "grid_ca{}{}.eps".format(refx, refy)
    plt.savefig(name, bbox_inches="tight")
    plt.show()


# plot_ccfs_grid()

ds_short = ds.isel(x=slice(-4, None))
refx = refx - 5

# Extract the reference time series
s_ref = ds_short.frames.isel(
    x=refx, y=refy
).values  # Select the time series at (refx, refy)
tau, _ = fpp.corr_fun(
    ds_short.frames.isel(x=0, y=0).values, s_ref, dt=5e-7
)  # Apply correlation function to each time series


def get_2d_corr(x, y):
    ref_signal = ds_short.frames.isel(x=x, y=y).values  # Select the time series at (refx, refy)
    def corr_wrapper(s):
        tau, res = fpp.corr_fun(
            ref_signal, s, dt=5e-7
        )  # Apply correlation function to each time series
        return res

    ds_corr = xr.apply_ufunc(
        corr_wrapper,
        ds_short,
        input_core_dims=[["time"]],  # Each function call operates on a single time series
        output_core_dims=[["tau"]],  # Output is also a time array
        vectorize=True,
    )
    ds_corr = ds_corr.assign_coords(tau=tau)
    trajectory_times = tau[np.abs(tau) < 1e-5]
    return ds_corr.sel(tau=trajectory_times)


def rotated_blob(params, rx, ry, x, y):
    lx, ly, t = params
    xt = (x - rx) * np.cos(t) + (y - ry) * np.sin(t)
    yt = (y - ry) * np.cos(t) - (x - rx) * np.sin(t)
    return np.exp(-((xt / lx) ** 2) - ((yt / ly) ** 2))


def plot_2d_ccf(x, y, ax):
    corr_data = get_2d_corr(x, y)
    rx, ry = corr_data.R.isel(x=x, y=y).values, corr_data.Z.isel(x=x,y=y).values
    data = corr_data.sel(tau=0).frames.values

    def model(params):
        blob = rotated_blob(params, rx, ry, corr_data.R.values, corr_data.Z.values)
        return np.sum((blob - data) ** 2)

    # Initial guesses for lx, ly, and t
    # Rough estimation
    bounds = [
        (0, 5),  # lx: 0 to 5
        (0, 5),  # ly: 0 to 5
        (-np.pi / 4, np.pi / 4)  # t: 0 to 2π
    ]

    result = differential_evolution(
        model,
        bounds,
        seed=42,  # Optional: for reproducibility
        popsize=15,  # Optional: population size multiplier
        maxiter=1000  # Optional: maximum number of iterations
    )

    im = ax.imshow(corr_data.sel(tau=0).frames, origin="lower", interpolation="spline16")
    ax.scatter(rx, ry, color="black")

    rmin, rmax, zmin, zmax = corr_data.R[0, 0] - 0.05, corr_data.R[0, -1], corr_data.Z[0, 0], corr_data.Z[-1, 0]

    def ellipse_parameters(params, alpha):
        lx, ly, t = params
        lx, ly = lx / 2, ly / 2
        xvals = lx * np.cos(alpha) * np.cos(t) - ly * np.sin(alpha) * np.sin(t) + rx
        yvals = lx * np.cos(alpha) * np.sin(t) + ly * np.sin(alpha) * np.cos(t) + ry
        return xvals, yvals

    alphas = np.linspace(0, 2 * np.pi, 200)
    elipsx, elipsy = zip(*[ellipse_parameters(result.x, a) for a in alphas])
    ax.plot(elipsx, elipsy)
    im.set_extent((rmin, rmax, zmin, zmax))

    return result.x


# Define the function to fit
fig, ax = plt.subplots(3, 5, figsize=(30, 50))
yvals = [7, 5, 4, 3, 1]
xvals = [1, 2, 3]

for x in range(3):
    for y in range(5):
        lx, ly, t = plot_2d_ccf(xvals[x], yvals[y], ax[x, y])
        ax[x, y].set_title(r"lx = {:.2f}, ly = {:.2f}, a = {:.2f}".format(lx, ly, t))

plt.savefig("2d_ccf_{}{}".format(refx+5, refy), bbox_inches="tight")
plt.show()

ds_corr = get_2d_corr(refx, refy)

pd = ve.estimate_velocities_for_pixel(refx, refy, ve.CModImagingDataInterface(ds_short), estimation_options=eo)
vx, vy = pd.vx, pd.vy
deltax = ds_corr.R.isel(x=refx, y=refy).values - ds_corr.R.isel(x=refx-1, y=refy).values
deltay = ds_corr.Z.isel(x=refx, y=refy).values - ds_corr.Z.isel(x=refx, y=refy-1).values

u2 = vx ** 2 + vy ** 2
taux = (vx * deltax) / u2
tauy = (vy * deltay) / u2

def get_taumax(v, w, dx, dy):
    lx_fit, ly_fit = 0.518, 1.686
    t_fit = -0.35
    a1 = (dx * ly_fit ** 2 * v + dy * lx_fit ** 2 * w) * np.cos(t_fit) ** 2
    a2 = (lx_fit ** 2 - ly_fit ** 2) * (dy * v + dx * w) * np.cos(t_fit) * np.sin(t_fit)
    a3 = (dx * lx_fit ** 2 * v + dy * ly_fit ** 2 * w) * np.sin(t_fit) ** 2
    d1 = (ly_fit ** 2 * v ** 2 + lx_fit ** 2 * w ** 2) * np.cos(t_fit) ** 2
    d2 = (lx_fit ** 2 * v ** 2 + ly_fit ** 2 * w ** 2) * np.sin(t_fit) ** 2
    d3 = 2 * (lx_fit ** 2 - ly_fit ** 2) * v * w * np.cos(t_fit) * np.sin(t_fit)
    return (a1 - a2 + a3) / (d1 + d2 - d3)


def error_tau(params):
    vx, vy = params
    taux_given = get_taumax(vx, vy, deltax, 0)  # Calculate tau_x
    tauy_given = get_taumax(vx, vy, 0, deltay)  # Calculate tau_y
    return (taux_given - taux) ** 2 + (tauy_given - tauy) ** 2


bonds = [(-5e6, 5e6), (-5e6, 5e6)]
result_u = differential_evolution(
    error_tau,
    bonds,
    seed=42,  # Optional: for reproducibility
    popsize=15,  # Optional: population size multiplier
    maxiter=1000  # Optional: maximum number of iterations
)


print(result_u)



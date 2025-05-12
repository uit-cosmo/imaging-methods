from phantom.show_data import *
from phantom.utils import *
from velocity_estimation.correlation import corr_fun
from scipy.optimize import minimize
from scipy import signal

shot = 1160616025

# ds = get_sample_data(shot, 0.005)
ds = xr.open_dataset("data.nc")
refx, refy = 6, 5

data_series = ds.frames.isel(x=refx, y=refy).values

cutoff_freq = 1e8


def get_pds_fit(freqs, taud, lamda):
    l = 1 / (1 + lamda**2)
    return (
        4
        * taud
        / ((1 + (1 - l) ** 2 * (taud * freqs) ** 2) * (1 + (l * taud * freqs) ** 2))
    )


def get_error_pdf_fit(params, expected):
    return np.sum(
        (
            get_pds_fit(freqs[freqs < cutoff_freq], params[0], params[1])
            - expected[freqs < cutoff_freq]
        )
        ** 2
    )


freqs, psd = signal.welch(data_series, fs=1 / get_dt(ds), nperseg=10**4)
freqs = 2 * np.pi * freqs

minimization = minimize(
    lambda params: get_error_pdf_fit(params, psd),
    [10, 1],
    method="Nelder-Mead",
    options={"maxiter": 1000},
)
lam = minimization.x[1]
tau = minimization.x[0]

fig, ax = plt.subplots()
ax.plot(freqs, psd)
ax.plot(freqs, get_pds_fit(freqs, tau, lam), ls="--", label="fit")
ax.text(
    0.1,
    0.8,
    r"$\tau_d: {:.2g}$".format(tau),
    transform=ax.transAxes,
    fontsize=10,
    alpha=0.5,
)
ax.text(
    0.1,
    0.7,
    r"$\lambda: {:.2g}$".format(1 / (1 + lam**2)),
    transform=ax.transAxes,
    fontsize=10,
    alpha=0.5,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e0, 1e7)

plt.show()

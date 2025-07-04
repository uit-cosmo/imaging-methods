import numpy as np

from phantom.utils import *
import matplotlib.pyplot as plt
from phantom.contours import *
import phantom as ph
from scipy import signal as s
import fppanalysis as fppa

shot = 1160616009
manager = ph.PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot)

refx, refy = 6, 6
dt = 5e-7
nperseg = 2e3
cutoff = 1e-4
taud_other = 1e-5
lam_other = 0.1

signal = ds.frames.isel(x=refx, y=refy).values

fig, ax = plt.subplots(1, 2)

taud, lam, times = ph.DurationTimeEstimator(
    ph.SecondOrderStatistic.ACF, ph.Analytics.TwoSided
).plot_and_fit(signal, 5e-7, ax[0], cutoff=cutoff)

label = (rf"$\tau_d = {taud_other:.2g}\, \lambda = {lam_other:.2g}$",)
ax[0].plot(
    times,
    ph.autocorrelation(times, taud_other, lam_other),
    ls="--",
    color="black",
    label=label,
)
ax[0].legend()

base, values = fppa.corr_fun(signal, fs=1 / dt, nperseg=nperseg)
base = 2 * np.pi * base  # Convert to angular frequency (rad/s)

mask = np.abs(base) < cutoff
mask[0] = False
base, values = base[mask], values[mask]


def obj_fun(params, base, expected):
    # untransformed_params = [params[0], 1 / (1 + params[1] ** 2)]
    analytical = ph.power_spectral_density(base, params[0], params[1])
    return (analytical - expected) ** 2


errors_fit = obj_fun([taud, lam], times, values)
errors_other = obj_fun([taud_other, lam_other], times, values)

ax[1].scatter(times, errors_fit, label="{:.2E}".format(np.sum(errors_fit)))
ax[1].scatter(
    times, errors_other, color="black", label="{:.2E}".format(np.sum(errors_other))
)
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].legend()
print(taud * 1.2)

plt.show()

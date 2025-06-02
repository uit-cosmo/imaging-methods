from phantom.show_data import *
import phantom as ph
from realizations import make_2d_realization
from blobmodel import BlobShapeEnum, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np
import cosmoplots as cp
import superposedpulses as sp

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

num_blobs = 50000
T = 20000
Lx = 10
Ly = 10
aspect_ratio = 1  # lx/ly
lx = np.sqrt(aspect_ratio)
ly = 1 / np.sqrt(aspect_ratio)
nx = 8
ny = 8
dt = 0.1
vx = 1
vy = 0
theta = -np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian)

run_norm_radius = 1000

refx, refy = 4, 4

size_scan = True


def get_realization(duration, T=1e4, noise=0):
    my_forcing_gen = sp.StandardForcingGenerator()
    my_forcing_gen.set_duration_distribution(lambda k: duration * np.ones(k))

    pm = sp.PointModel(waiting_time=1, total_duration=T, dt=dt)
    pm.set_pulse_shape(
        sp.ExponentialShortPulseGenerator(lam=0, tolerance=1e-20, max_cutoff=T)
    )

    pm.set_custom_forcing_generator(my_forcing_gen)

    times, signal = pm.make_realization()

    if noise != 0:
        signal = signal + np.random.normal(0, noise, len(signal))

    signal = (signal - signal.mean()) / signal.std()

    return times, signal


def make_realization_and_estimate_duration_time(duration, T):
    times, signal = get_realization(duration, T)
    taud, _ = ph.DurationTimeEstimator(
        ph.SecondOrderStatistic.ACF, ph.Analytics.TwoSided
    ).estimate_duration_time(
        signal,
        dt,
        cutoff=None,
        nperseg=1000,
    )
    return taud


def run_realizations_and_estimate_bias_and_std(duration, T, num_realizations=10):
    tauds = np.array(
        [
            make_realization_and_estimate_duration_time(duration, T)
            for _ in np.arange(num_realizations)
        ]
    )
    return (tauds.mean() - duration) / duration, tauds.std()


fromm = -2
to = 2
parameters = np.logspace(fromm, to, num=6)

results = np.array(
    [run_realizations_and_estimate_bias_and_std(p, 1e3) for p in parameters]
)
results = np.vstack(results)

fig, ax = cp.figure_multiple_rows_columns(1, 1)
ax = ax[0]
ax.scatter(
    parameters / dt, results[:, 0], color="blue", label=r"$(\widehat{\tau}-\tau)/\tau$"
)
ax.scatter(parameters / dt, results[:, 1], color="green", label=r"$\sigma_\tau$")

ax.legend()
ax.set_xlabel(r"$\tau_\text{d}/dt$")

ax.set_xscale("log")
plt.savefig("taud_estimate.png", bbox_inches="tight")
plt.show()

quit()


tauds = np.logspace(-1, 1, num=10)
if size_scan:
    size_factors = np.logspace(-1, 1, num=10)
    tauds = np.zeros_like(size_factors)
    for i in range(len(size_factors)):
        s = size_factors[i]

        print("Input duration {:.2f}, estimated duration {:.2f}".format(s, taud))
        tauds[i] = taud

quit()
s = 1

use_blobmodel = True
if use_blobmodel:
    ds = make_2d_realization(
        Lx, Ly, T, nx, ny, dt, num_blobs, vx, vy, s * lx, s * ly, theta, bs
    )
    ds = ph.run_norm_ds(ds, run_norm_radius)

    fig, ax = cp.figure_multiple_rows_columns(1, 1)
    ax = ax[0]
    taud, lam, freqs = ph.DurationTimeEstimator(
        ph.SecondOrderStatistic.PSD, ph.Analytics.TwoSided
    ).plot_and_fit(
        ds.frames.isel(x=refx, y=refy).values,
        dt,
        ax=ax,
        cutoff=10000,
        nperseg=1000,
    )

    plt.show()

use_point_model = False
if use_point_model:
    times, signal = get_realization(1)

    fig, ax = cp.figure_multiple_rows_columns(1, 1)
    ax = ax[0]
    taud, lam, freqs = ph.DurationTimeEstimator(
        ph.SecondOrderStatistic.PSD, ph.Analytics.TwoSided
    ).plot_and_fit(
        signal,
        dt,
        ax=ax,
        cutoff=10000,
        nperseg=1000,
    )

    plt.show()

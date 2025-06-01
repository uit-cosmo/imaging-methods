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

size_scan = False


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


if size_scan:
    size_factors = np.logspace(-1, 1, num=10)
    tauds = np.zeros_like(size_factors)
    for i in range(len(size_factors)):
        s = size_factors[i]
        print(s)
        ds = make_2d_realization(
            Lx, Ly, T, nx, ny, dt, num_blobs, vx, vy, s * lx, s * ly, theta, bs
        )
        ds = ph.run_norm_ds(ds, run_norm_radius)

        taud, lam = ph.DurationTimeEstimator(
            ph.SecondOrderStatistic.ACF, ph.Analytics.TwoSided
        ).estimate_duration_time(
            ds.frames.isel(x=refx, y=refy).values,
            get_dt(ds),
            cutoff=100,
            nperseg=1000,
        )
        tauds[i] = taud

    print(tauds)

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

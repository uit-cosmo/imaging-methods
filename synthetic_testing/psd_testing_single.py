from phantom import power_spectral_density
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

dt = 0.1


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


times, signal = get_realization(100, 1e3)

fig, ax = cp.figure_multiple_rows_columns(1, 1)
ax = ax[0]

taud, _, freqs = ph.DurationTimeEstimator(
    ph.SecondOrderStatistic.PSD, ph.Analytics.TwoSided
).plot_and_fit(
    signal,
    dt,
    ax,
    cutoff=None,
    nperseg=1000,
)

ax.plot(freqs, power_spectral_density(freqs, 100, 0), ls="--")

plt.show()

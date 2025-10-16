import blobmodel as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import welch
import fppanalysis as fppa
from scipy.stats import skew, kurtosis

import cosmoplots as cp

# matplot_params = plt.rcParams
# cp.set_rcparams_dynamo(matplot_params, 2)
# plt.rcParams.update(matplot_params)

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)


taup = 1
times = np.arange(-3, 3, step=0.01)


def gaussian(taup):
    return np.exp(-times / taup) * np.exp(-(times**2))


def one_exp(taup):
    return np.heaviside(times, 0) * np.exp(-times / taup) * np.exp(-times)


fig, ax = plt.subplots()

ax.plot(
    times,
    one_exp(taup),
    color="black",
    label=r"$\frac{v}{\ell}\tau_{\shortparallel}=1$",
)
ax.plot(times, gaussian(taup), color="black")

ax.plot(
    times, one_exp(10), color="blue", label=r"$\frac{v}{\ell}\tau_{\shortparallel}=10$"
)
ax.plot(times, gaussian(10), color="blue")

ax.set_xlabel(r"$\frac{v}{\ell}(t-t_k)$")
ax.legend()

plt.show()

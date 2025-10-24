import blobmodel as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import welch
import fppanalysis as fppa
from scipy.stats import skew, kurtosis

import cosmoplots as cp

matplot_params = plt.rcParams
cp.set_rcparams_dynamo(matplot_params, 2)
plt.rcParams.update(matplot_params)

# plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)


taup = 5
times = np.arange(-1, 4, step=0.01)


def gaussian(taup, x):
    return np.exp(-times / taup) * np.exp(-((x - times) ** 2))


def one_exp(taup, x):
    return np.heaviside(times - x, 0) * np.exp(-times / taup) * np.exp(x - times)


fig, ax = plt.subplots(1, 2)


def process_ax(ax, f):
    ax.plot(
        times,
        f(taup, 0),
        color="black",
        label=r"$\frac{v}{\ell}\tau_{\shortparallel}=5$",
    )

    ax.plot(times, f(taup, 1), color="black")
    ax.plot(
        times,
        f(1e10, 0),
        color="blue",
        label=r"$\frac{v}{\ell}\tau_{\shortparallel}=\infty$",
    )
    ax.plot(times, f(1e10, 1), color="blue")

    ax.set_xlabel(r"$\frac{v}{\ell}(t-t_k)$")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$0$", r"$\frac{\varphi(t-t_k)}{\varphi_0}$"])
    ax.set_ylim([0, 1.2])
    ax.text(0 - 0.25, 1.07, r"$x=0$")
    ax.text(1 - 0.25, 1.07, r"$x=\ell$")
    ax.legend()


process_ax(ax[0], one_exp)
process_ax(ax[1], gaussian)

plt.savefig("2d_fpp_effective_ps.pdf", bbox_inches="tight")

plt.show()

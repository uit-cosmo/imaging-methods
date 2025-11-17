import numpy as np

from decoherence_utils import *
import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

data_file = "decoherence_data.npz"

i = 0
N = 5

# --------------------------------------------------------------
# 2.  SWEEP OVER theta
# --------------------------------------------------------------
rand_coeffs = np.linspace(0, 1, num=10)

loaded = np.load(data_file)
rand_coeffs = loaded["rand_coeffs"]
vx_all = loaded["vx_all"]
vy_all = loaded["vy_all"]
vxtde_all = loaded["vxtde_all"]
vytde_all = loaded["vytde_all"]
confidences = loaded["confidences"]
vx_2dcas = loaded["vx_2dcas"]
vy_2dcas = loaded["vy_2dcas"]
cond_reprs = loaded["cond_reprs"]


# --------------------------------------------------------------
# 3.  SCATTER PLOT
# --------------------------------------------------------------
fig, ax = plt.subplots()


# Helper: scatter one component
def scatter_component(vals, data_per_theta, label, marker, color):
    first = True
    for th, vals in zip(vals, data_per_theta):
        jitter = 0  # np.random.normal(0, 0.003, size=len(vals))
        ax.scatter(
            np.full_like(vals, th) + jitter,
            vals,
            marker=marker,
            s=40,
            edgecolor="k",
            linewidth=0.3,
            label=label if first else None,  # <-- only label first
            alpha=0.7,
            color=color,
        )
        first = False


# Choose distinct colours (you can also use a colormap)
scatter_component(rand_coeffs, vx_all, r"$v_c$", "o", "#1f77b4")
# scatter_component(rand_coeffs, vy_all, r"$w_c$", "s", "#ff7f0e")
scatter_component(rand_coeffs, vx_2dcas, r"$v_{\mathrm{2DCA TDE}}$", "^", "#2ca02c")
# scatter_component(rand_coeffs, vy_2dcas, r"$w_{\mathrm{2DCA TDE}}$", "d", "#d62728")
scatter_component(rand_coeffs, vxtde_all, r"$v_{\mathrm{TDE}}$", "v", "#2ca02c")
# scatter_component(rand_coeffs, vytde_all, r"$w_{\mathrm{TDE}}$", "p", "#d62728")

ax.set_xlabel(r"$r$")
ax.set_ylabel("Velocity estimates")
ax.legend()  # loc=6
# ax.set_xlim(0, 1.2)
ax.set_ylim(-0.2, 2)

plt.savefig("decoherence_scan.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()

scatter_component(rand_coeffs, confidences, r"$\text{Conf.}$", "s", "blue")
scatter_component(rand_coeffs, cond_reprs, r"$\text{Cond Repr.}$", "^", "red")

ax.legend()
ax.set_xlabel(r"$r$")

plt.savefig("decoherence_scan_conf.pdf", bbox_inches="tight")
plt.show()

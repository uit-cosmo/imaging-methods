import numpy as np

from ..utils import *
from .decoherence_utils import *
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
force_redo = False

i = 0





T = 1000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 1000

vx_input = 1
vy_intput = 0
lx_input = 1
ly_input = 1
N = 3


def get_all_velocities(lx, ly, rand_coeff, N=N):
    """
    Run N realisations and return the *raw* velocity components.

    Returns
    -------
    vx_all, vy_all, vxtde_all, vytde_all : list (length N)
        One entry per Monte-Carlo realisation.
    """
    vx_all = []
    vy_all = []
    vx_cc_tde_all = []
    vy_cc_tde_all = []
    confidence_all = []
    vx_2dca_tde_all = []
    vy_2dca_tde_all = []
    cond_repr_all = []

    def blob_getter():
        alpha = np.random.uniform(-np.pi/2 * rand_coeff, np.pi/2 * rand_coeff)
        u = np.random.uniform(1-rand_coeff, 1+rand_coeff)
        return get_blob(
            amplitude=np.random.exponential(),
            vx=u * np.cos(alpha),
            vy=u * np.sin(alpha),
            posx=0,
            posy=np.random.uniform(0, Ly),
            lx=1,
            ly=1,
            t_init=np.random.uniform(0, T),
            bs=bs,
            theta=0,
        )

    for _ in range(N):
        ds = make_2d_realization(
            Lx,
            Ly,
            T,
            nx,
            ny,
            dt,
            K,
            vx=vx_input,
            vy=vy_intput,
            lx=lx,
            ly=ly,
            theta=0,
            bs=bs,
            blob_getter=blob_getter,
        )
        ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])

        v_c, w_c, vx_cc_tde, vy_cc_tde, confidence, vx_2dca_tde, vy_2dca_tde, cond_repr = estimate_velocities(ds, method_parameters)

        vx_all.append(v_c)
        vy_all.append(w_c)
        vx_cc_tde_all.append(vx_cc_tde)
        vy_cc_tde_all.append(vy_cc_tde)
        confidence_all.append(confidence)
        vx_2dca_tde_all.append(vx_2dca_tde)
        vy_2dca_tde_all.append(vy_2dca_tde)
        cond_repr_all.append(cond_repr)

    return vx_all, vy_all, vx_cc_tde_all, vy_cc_tde_all, confidence_all, vx_2dca_tde_all, vy_2dca_tde_all, cond_repr_all


# --------------------------------------------------------------
# 2.  SWEEP OVER theta
# --------------------------------------------------------------
lx, ly = 0.5, 2.0
rand_coeffs = np.linspace(0, 1, num=10)

# Containers for *all* realisations (list of lists)
vx_all = []  # len = len(thetas); each entry = list of N values
vy_all = []
vxtde_all = []
vytde_all = []
confidences = []
vx_2dcas = []
vy_2dcas = []
cond_reprs = []

if os.path.exists(data_file) and not force_redo:
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
else:
    for rand_coeff in rand_coeffs:
        print(f"Processing rand coeff = {rand_coeff:.3f}")
        vx, vy, vx_cc_tde_all, vy_cc_tde_all, confidence_all, vx_2dca_tde_all, vy_2dca_tde_all, cond_repr_all = get_all_velocities(lx, ly, rand_coeff, N=N)

        vx_all.append(vx)
        vy_all.append(vy)
        vxtde_all.append(vx_cc_tde_all)
        vytde_all.append(vy_cc_tde_all)
        confidences.append(confidence_all)
        vx_2dcas.append(vx_2dca_tde_all)
        vy_2dcas.append(vy_2dca_tde_all)
        cond_reprs.append(cond_repr_all)

    np.savez(
        data_file,
        rand_coeffs=rand_coeffs,
        vx_all=vx_all,
        vy_all=vy_all,
        vxtde_all=vxtde_all,
        vytde_all=vytde_all,
        confidences=confidences,
        vx_2dcas=vx_2dcas,
        vy_2dcas=vy_2dcas,
        cond_reprs=cond_reprs,
    )

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
scatter_component(rand_coeffs, vy_all, r"$w_c$", "s", "#ff7f0e")
scatter_component(rand_coeffs, vxtde_all, r"$v_{\mathrm{TDE}}$", "^", "#2ca02c")
scatter_component(rand_coeffs, vytde_all, r"$w_{\mathrm{TDE}}$", "d", "#d62728")

ax.set_xlabel(r"$r$")
ax.set_ylabel("Velocity estimates")
ax.legend()  # loc=6

plt.savefig("decoherence_scan.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()

scatter_component(rand_coeffs, confidences, r"$\text{Conf.}$", "s", "blue")
scatter_component(rand_coeffs, cond_reprs, r"$\text{Cond Repr.}$", "^", "red")

ax.legend()
ax.set_xlabel(r"$r$")

plt.savefig("decoherence_scan_conf.pdf", bbox_inches="tight")
plt.show()



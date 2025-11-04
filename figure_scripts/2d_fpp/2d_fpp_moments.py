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

dt = 0.1
T = int(1e5)
num_blobs = int(1e5)
start_index = int(100 / dt)
Lx, Ly = 10, 10
rate = Ly * T / num_blobs


def get_realization(shape_x, taup, vx=1, vy=0):
    # Blobmodel
    model = bm.Model(
        Nx=1,
        Ny=1,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        T=T,
        num_blobs=num_blobs,
        blob_shape=bm.BlobShapeImpl(shape_x, bm.BlobShapeEnum.gaussian),
        periodic_y=False,
        t_drain=taup,
        blob_factory=bm.DefaultBlobFactory(
            A_dist=bm.DistributionEnum.exp,
            vy_dist=bm.DistributionEnum.deg,
            vy_parameter=vy,
            vx_parameter=vx,
        ),
        verbose=True,
        t_init=0,
        one_dimensional=False,  # Sets Ly = 0, Ny = 1, and checks vy = 0, and sets the y blob shape to 1.
    )
    x_matrix, y_matrix, t_matrix = np.meshgrid(
        model._geometry.x, [5.0], model._geometry.t
    )
    model._geometry.y_matrix = y_matrix
    ds = model.make_realization(speed_up=True, error=1e-10)
    ds = ds.isel(t=slice(start_index, int(1e50)))
    signal = ds.n.isel(x=0, y=0).values
    return signal


def get_mean(shape_x, taup):
    if shape_x == bm.BlobShapeEnum.exp:
        tau = taup / (1 + taup)
        return tau / rate
    if shape_x == bm.BlobShapeEnum.gaussian:
        return np.exp((1 / (2 * taup)) ** 2) / rate


def get_rfl(shape_x, taup):
    if shape_x == bm.BlobShapeEnum.exp:
        tau = taup / (1 + taup)
        return np.sqrt(rate / tau) / (2 * np.pi) ** (1 / 4)
    if shape_x == bm.BlobShapeEnum.gaussian:
        return np.sqrt(rate / np.pi)


def get_skew(shape_x, taup):
    if shape_x == bm.BlobShapeEnum.exp:
        tau = taup / (1 + taup)
        return np.sqrt(rate / tau) * 2 ** (7 / 4) / (3 * np.pi ** (1 / 2)) ** (1 / 2)
    if shape_x == bm.BlobShapeEnum.gaussian:
        return np.sqrt(4 * rate / np.pi)


def get_kurt(shape_x, taup):
    if shape_x == bm.BlobShapeEnum.exp:
        tau = taup / (1 + taup)
        return 6 * rate / (np.sqrt(np.pi) * tau)
    if shape_x == bm.BlobShapeEnum.gaussian:
        return 6 * rate / np.pi


def get_estimated_moments(shape_x, taup):
    signal = get_realization(shape_x, taup)
    mean = np.mean(signal)
    rfl = np.mean((signal - mean) ** 2) ** (1 / 2) / mean
    skewness = skew(signal)
    kurt = kurtosis(signal)
    return mean, rfl, skewness, kurt


taus = np.logspace(0, 2, num=5)
moments_exp = [get_estimated_moments(bm.BlobShapeEnum.exp, tp) for tp in taus]
moments_gauss = [get_estimated_moments(bm.BlobShapeEnum.gaussian, tp) for tp in taus]

means = [np.array([m[0] for m in moments_exp]), np.array([m[0] for m in moments_gauss])]
rfls = [np.array([m[1] for m in moments_exp]), np.array([m[1] for m in moments_gauss])]
skews = [np.array([m[2] for m in moments_exp]), np.array([m[2] for m in moments_gauss])]
kurts = [np.array([m[3] for m in moments_exp]), np.array([m[3] for m in moments_gauss])]

fig, ax = plt.subplots(2, 2)


def process_ax(ax, data, f, label):
    ax.scatter(taus, data[0], color="blue")
    ax.scatter(taus, data[1], color="black")
    analytical_exp = np.array([f(bm.BlobShapeEnum.exp, tp) for tp in taus])
    ax.scatter(taus, analytical_exp, color="blue", marker="x")
    analytical_gauss = np.array([f(bm.BlobShapeEnum.gaussian, tp) for tp in taus])
    ax.scatter(taus, analytical_gauss, color="black", marker="x")
    ax.set_xscale("log")
    ax.set_ylabel(label)
    ax.set_xlabel(r"$\tau_{\parallel}$")
    ax.set_ylim(0, 1.2 * np.max(data))


process_ax(ax[0][0], means, get_mean, r"$<\Phi>$")
process_ax(ax[0][1], rfls, get_rfl, r"$\Phi_{\text{rms}}/<\Phi>$")
process_ax(ax[1][0], skews, get_skew, r"$S_\Phi$")
process_ax(ax[1][1], kurts, get_kurt, r"$F_\Phi$")

plt.show()

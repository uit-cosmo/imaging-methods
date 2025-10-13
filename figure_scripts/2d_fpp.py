import blobmodel as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import welch
import fppanalysis as fppa

import cosmoplots as cp

matplotlib_params = plt.rcParams
cp.set_rcparams_dynamo(matplotlib_params, 2)
plt.rcParams.update(matplotlib_params)

dt = 0.1
T = int(1e5)
num_blobs = int(1e5)
start_index = int(100 / dt)
taup = 1e10


def get_realization(shape_x, taup, vx=1, vy=0):
    # Blobmodel
    model = bm.Model(
        Nx=1,
        Ny=1,
        Lx=1,
        Ly=0,
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
    ds = model.make_realization(speed_up=True, error=1e-10)
    ds = ds.isel(t=slice(start_index, int(1e50)))
    signal = ds.n.isel(x=0, y=0).values
    return signal


def get_analytical_acf(taus, shape, tau_p, w):
    if shape == bm.BlobShapeEnum.exp:
        tau = tau_p / (1 + np.sqrt(1 + w**2) * tau_p)  # ell = 1, v = 1
        return np.exp(-np.abs(taus) / tau)
    if shape == bm.BlobShapeEnum.gaussian:
        tau = 1 / np.sqrt(1 + w**2)
        return np.exp(-1 / 2 * (taus / tau) ** 2)
    return np.zeros(len(taus))


def plot_synthetic_data_acf(
    shape_x,
    taup,
    w,
    ax,
    color=None,
    label=None,
    plot_analytical=True,
):
    signal = get_realization(shape_x, taup, vx=1, vy=w)
    taus, acf = fppa.corr_fun(signal, signal, dt)
    window_size = 3
    mask = np.abs(taus) < window_size
    taus, acf = taus[mask], acf[mask]
    ax.plot(taus, acf, color=color, label=label, ls="-")

    if plot_analytical:
        ax.plot(taus, get_analytical_acf(taus, shape_x, taup, w), color=color, ls="--")


fig, ax = plt.subplots(1, 2)

plot_synthetic_data_acf(
    bm.BlobShapeEnum.exp, 1e10, 0, ax[0], color="blue", label=r"No damping $w/v=0$"
)
plot_synthetic_data_acf(
    bm.BlobShapeEnum.exp, 1, 0, ax[0], color="red", label=r"Damping $w/v=0$"
)
plot_synthetic_data_acf(
    bm.BlobShapeEnum.exp, 1e10, 1, ax[0], color="green", label=r"No Damping $w/v=1$"
)
plot_synthetic_data_acf(
    bm.BlobShapeEnum.exp, 1, 1, ax[0], color="black", label=r"Damping $w/v=1$"
)
# ax[0].legend()
ax[0].set_ylabel(r"$R_{\tilde{\Phi}}(\triangle_t)$")
ax[0].set_xlabel(r"$\frac{v}{\ell}\triangle_t$")
ax[0].set_title("Exp shape")

plot_synthetic_data_acf(
    bm.BlobShapeEnum.gaussian, 1e10, 0, ax[1], color="blue", label=r"No damping $w/v=0$"
)
plot_synthetic_data_acf(
    bm.BlobShapeEnum.gaussian, 1, 0, ax[1], color="red", label=r"Damping $w/v=0$"
)
plot_synthetic_data_acf(
    bm.BlobShapeEnum.gaussian,
    1e10,
    1,
    ax[1],
    color="green",
    label=r"No Damping $w/v=1$",
)
plot_synthetic_data_acf(
    bm.BlobShapeEnum.gaussian, 1, 1, ax[1], color="black", label=r"Damping $w/v=1$"
)
ax[1].legend()
ax[1].set_ylabel(r"$R_{\tilde{\Phi}}(\triangle_t)$")
ax[1].set_xlabel(r"$\frac{v}{\ell}\triangle_t$")
ax[1].set_title("Gaussian shape")

plt.savefig("2d_fpp_acf_test.pdf", bbox_inches="tight")
plt.show()

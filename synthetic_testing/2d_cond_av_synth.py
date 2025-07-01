from phantom.show_data import *
from phantom.cond_av import *
from phantom.contours import *
from phantom.parameter_estimation import *
import phantom as ph
from realizations import make_2d_realization
from blobmodel import BlobShapeEnum, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np
import cosmoplots as cp

from synthetic_testing.realizations import BlobParameters

num_blobs = 500
T = 2000
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
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)


def estimate_blob_parameters(ds, id):
    # preprocessing
    # ds = ph.run_norm_ds(ds, 1000)

    # 2DCA
    refx, refy = 4, 4
    events, average, std = find_events_and_2dca(
        ds, refx, refy, threshold=0.2, check_max=1, window_size=30, single_counting=True
    )

    # Gaussian Fit
    v_f, w_f = get_3tde_velocities(average)
    fig, ax = plt.subplots()
    lx, ly, theta = plot_event_with_fit(average, ax, "2dca_fit_{}.png".format(id))
    fig.clf()

    # Contouring
    contours_ds = get_contour_evolution(average, threshold_factor=0.5)
    velocity_ds = get_contour_velocity(contours_ds.center_of_mass, window_size=11)

    show_movie_with_contours(
        average,
        refx,
        refy,
        contours_ds,
        "frames",
        lims=(0, 0.3),
        gif_name="2dca_contour_{}.gif".format(id),
        interpolation=None,
        show=False,
    )

    fig, ax = cp.figure_multiple_rows_columns(1, 1)
    ax = ax[0]

    ax.plot(velocity_ds.time, velocity_ds.values[:, 0], label=r"$v_c$")
    tmin, tmax = velocity_ds.time.max().item(), velocity_ds.time.min().item()
    ax.hlines(vx, tmin, tmax, label=r"$v$", ls="--")
    ax.plot(velocity_ds.time, velocity_ds.values[:, 1], label=r"$w_c$")
    ax.hlines(vy, tmin, tmax, label=r"$w$", ls="--")
    ax.legend()
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$v, w$")
    fig.clf()

    vx_c, vy_c = velocity_ds.sel(time=0).values
    area = contours_ds.area_c.sel(time=0).item()
    size = np.sqrt(area)

    # PSD

    fig, ax = cp.figure_multiple_rows_columns(1, 1)
    ax = ax[0]
    taud, lam = ph.fit_psd(
        ds.frames.isel(x=refx, y=refy).values,
        get_dt(ds),
        nperseg=10**3,
        ax=ax,
        cutoff_freq=1e6,
    )
    plt.savefig("psd_fit_{}".format(id), bbox_inches="tight")
    fig.clf()

    bp = BlobParameters(vx_c, vy_c, area, vx, vy, lx, ly, theta, taud, lam)
    return bp


ds = make_2d_realization(Lx, Ly, T, nx, ny, dt, num_blobs, vx, vy, lx, ly, theta, bs)
# show_movie(ds.sel(time=slice(10, 20)), lims=(0, 0.3))
bp = estimate_blob_parameters(ds, 0)
# print(bp)

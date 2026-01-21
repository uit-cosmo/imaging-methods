import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import os
import cosmoplots as cp
import xarray as xr
import numpy as np

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)


mp = im.get_default_synthetic_method_params()
mp.position_filter.window_size = 11

T = 5000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 5000

NSR = 0.1

lx, ly = 0.5, 2.0
theta = 1.57
i = 3

def get_simulation_data(lx, ly, theta):
    vx_input, vy_input = 1, 0
    ds = im.make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_input,
        lx=lx,
        ly=ly,
        theta=theta,
        bs=bs,
    )
    ds_mean = ds.frames.mean().item()
    ds = ds.assign(
        frames=ds["frames"] + ds_mean * NSR * np.random.random(ds.frames.shape)
    )
    ds = im.run_norm_ds(ds, mp.preprocessing.radius)
    ds["v_input"] = vx_input
    ds["w_input"] = vy_input
    return ds


def plot_frames_with_contour(average, contours, t_indexes, variable="cond_av"):
    fig, ax = plt.subplots(
        1, 3, figsize=(3 * 2.08, 1 * 2.08), gridspec_kw={"hspace": 0, "wspace": 0}
    )
    for i in np.arange(3):
        axe = ax[i]
        im = axe.imshow(
            average[variable].isel(time=int(t_indexes[i])).values,
            origin="lower",
            interpolation=None,  # "spline16",
        )
        c = contours.contours.isel(time=int(t_indexes[i])).data
        axe.plot(c[:, 0], c[:, 1], ls="--", color="black")
        im.set_extent(
            (average.R[0, 0], average.R[0, -1], average.Z[0, 0], average.Z[-1, 0])
        )
        im.set_clim(0, 2)
        axe.set_ylim((average.Z[0, 0], average.Z[-1, 0]))
        axe.set_xlim((average.R[0, 0], average.R[0, -1]))
        t = average["time"].isel(time=int(t_indexes[i])).item()
        axe.text(3, 6.5, r"$t={:.1f}\,\tau_\text{{d}}$".format(t), color="white")
        axe.set_xticks([])
        axe.set_yticks([])

    return fig


file_name = "synthetic_blob_2dca.nc"
#ds = get_simulation_data(0.5, 2, np.pi/4)
#ds.to_netcdf(file_name)
ds = xr.open_dataset(file_name)

dt = im.get_dt(ds)
dr = im.get_dr(ds)

tdca_params = mp.two_dca

events, average_ds = im.find_events_and_2dca(ds, mp.two_dca)

contour_ca = im.get_contour_evolution(
    average_ds.cond_av,
    mp.contouring.threshold_factor,
    max_displacement_threshold=None,
)
t_indexes = np.linspace(10, tdca_params.window-10, num=3)

plot_frames_with_contour(average_ds, contour_ca, t_indexes)

plt.savefig("synthetic_blob_2dca.eps", bbox_inches="tight")




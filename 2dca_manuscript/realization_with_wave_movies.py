import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import cosmoplots as cp
import numpy as np
from imaging_methods import movie_dataset, show_movie_with_contours
from scan_utils import *
import xarray as xr

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

method_parameters = im.get_default_synthetic_method_params()

T = 5000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 5000
N = 1
NSR = 0.1

wave_amplitude = 0.25
wavelength = 8
period = 8
vx_input, vy_input = 1, 0


def get_realization_with_wave():
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
        lx=1,
        ly=1,
        theta=0,
        bs=bs,
    )
    ds_mean = ds.frames.mean().item()
    ds = ds.assign(
        frames=ds["frames"] + ds_mean * NSR * np.random.random(ds.frames.shape)
    )

    k = 2 * np.pi / wavelength  # Wave number
    omega = 2 * np.pi / period  # Angular frequency
    wave = wave_amplitude * np.sin(k * ds.Z - omega * ds.time)  # Shape: (time, x)

    ds["frames"] = ds["frames"] + wave
    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    return ds


file_name = "realization_with_wave.nc"
# ds = get_realization_with_wave()
# ds.to_netcdf(file_name)
ds = xr.open_dataset(file_name)

movie_dataset(
    ds.sel(time=slice(500, 520)),
    gif_name="realization_with_wave.gif",
    show=False,
    lims=(0, 1),
    normalize=True,
    interpolation=None,
)

events, average_ds = im.find_events_and_2dca(ds, method_parameters.two_dca)

contour_ca = im.get_contour_evolution(
    average_ds.cond_av,
    method_parameters.contouring.threshold_factor,
    max_displacement_threshold=None,
)
contour_cc = im.get_contour_evolution(
    average_ds.cross_corr,
    method_parameters.contouring.threshold_factor,
    max_displacement_threshold=None,
)

show_movie_with_contours(
    average_ds,
    contour_ca,
    variable="cond_av",
    lims=(0, 1),
    gif_name="realization_with_wave_2dca_contour.gif",
    interpolation=None,
    show=False,
)

show_movie_with_contours(
    average_ds,
    contour_cc,
    variable="cross_corr",
    lims=(0, average_ds.cross_corr.max().item()),
    gif_name="realization_with_wave_2dcc_contour.gif",
    interpolation=None,
    show=False,
)


v_2dca, w_2dca = im.get_averaged_velocity(
    average_ds, "cond_av", method_parameters, position_method="contouring"
)
v_2dcc, w_2dcc = im.get_averaged_velocity(
    average_ds, "cross_corr", method_parameters, position_method="contouring"
)

pos_2dca, mask_2dca = get_positions_and_mask(
    average_ds, "cond_av", method_parameters, position_method="contouring"
)
pos_2dcc, mask_2dcc = get_positions_and_mask(
    average_ds, "cross_corr", method_parameters, position_method="contouring"
)

fig, axes = plt.subplots(1, 2)

v_input, w_input = vx_input, vy_input

print("CA velocities: {:.2f},   {:.2f}".format(v_2dca, w_2dca))
print("CC velocities: {:.2f},   {:.2f}".format(v_2dcc, w_2dcc))
print("Input velocities: {:.2f},   {:.2f}".format(v_input, w_input))


def plot_component(i, ax, position_ca, position_cc, mask_ca, mask_cc):
    input_velocity = v_input if i == 0 else w_input
    ax.plot(position_ca.time, position_ca.values[:, i], color="blue")
    ax.plot(position_cc.time, position_cc.values[:, i], color="green")
    ax.plot(
        position_ca.time,
        position_ca.sel(time=0).values[0] + position_ca.time.values * input_velocity,
        color="black",
        ls="--",
    )

    ax.plot(
        position_ca.time[mask_ca],
        position_ca.values[:, i][mask_ca],
        lw=2,
        color="blue",
    )
    ax.plot(
        position_cc.time[mask_cc],
        position_cc.values[:, i][mask_cc],
        lw=2,
        color="green",
    )


plot_component(0, axes[0], pos_2dca, pos_2dcc, mask_2dca, mask_2dcc)
plot_component(1, axes[1], pos_2dca, pos_2dcc, mask_2dca, mask_2dcc)

plt.show()

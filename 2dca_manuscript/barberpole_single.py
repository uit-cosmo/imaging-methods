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
    alpha = np.pi / 8
    vx_input, vy_input = np.cos(alpha), np.sin(alpha)
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


file_name = os.path.join("synthetic_data", "data_single.nc ")
# ds = get_simulation_data(0.5, 2, np.pi/4)
# ds.to_netcdf(file_name)
ds = xr.open_dataset(file_name)
plot_gifs = False


dt = im.get_dt(ds)
dr = im.get_dr(ds)

tdca_params = mp.two_dca
events, average_ds = im.find_events_and_2dca(ds, mp.two_dca)

contour_ca = im.get_contour_evolution(
    average_ds.cond_av,
    mp.contouring.threshold_factor,
    max_displacement_threshold=None,
)
contour_cc = im.get_contour_evolution(
    average_ds.cross_corr,
    mp.contouring.threshold_factor,
    max_displacement_threshold=None,
)

if plot_gifs:
    im.movie_dataset(
        ds.sel(time=slice(500, 510)), gif_name="barberpole_dataset.gif", show=False
    )
    im.show_movie_with_contours(
        average_ds,
        contour_ca,
        variable="cond_av",
        lims=(0, average_ds.cond_av.max().item()),
        gif_name="ca.gif",
        interpolation=None,
        show=False,
    )

    im.show_movie_with_contours(
        average_ds,
        contour_cc,
        variable="cross_corr",
        lims=(0, average_ds.cross_corr.max().item()),
        gif_name="cc.gif",
        interpolation=None,
        show=False,
    )


def get_positions_and_mask(
    average_ds,
    variable,
    method_parameters: im.MethodParameters,
    position_method="contouring",
):
    if position_method == "contouring":
        position_da = im.get_contour_evolution(
            average_ds[variable],
            method_parameters.contouring.threshold_factor,
            max_displacement_threshold=None,
        ).center_of_mass
    elif position_method == "max":
        position_da = im.compute_maximum_trajectory_da(
            average_ds, variable, method="parabolic"
        )
    else:
        raise NotImplementedError

    position_da, start, end = im.smooth_da(
        position_da, method_parameters.position_filter, return_start_end=True
    )

    mask = im.get_combined_mask(
        average_ds.isel(time=slice(start, end)),
        variable,
        position_da,
        method_parameters.position_filter,
    )

    return position_da, mask


v_2dca, w_2dca = im.get_averaged_velocity(
    average_ds, "cond_av", mp, position_method="contouring"
)
v_2dcc, w_2dcc = im.get_averaged_velocity(
    average_ds, "cross_corr", mp, position_method="contouring"
)
v_2dca_max, w_2dca_max = im.get_averaged_velocity(
    average_ds, "cond_av", mp, position_method="max"
)
v_2dcc_max, w_2dcc_max = im.get_averaged_velocity(
    average_ds, "cross_corr", mp, position_method="max"
)

pos_2dca, mask_2dca = get_positions_and_mask(
    average_ds, "cond_av", mp, position_method="contouring"
)
pos_2dcc, mask_2dcc = get_positions_and_mask(
    average_ds, "cross_corr", mp, position_method="contouring"
)
pos_max_2dca, mask_max_2dca = get_positions_and_mask(
    average_ds, "cond_av", mp, position_method="max"
)
pos_max_2dcc, mask_max_2dcc = get_positions_and_mask(
    average_ds, "cross_corr", mp, position_method="max"
)

fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharex=True, sharey=True)

v_input, w_input = ds["v_input"].item(), ds["w_input"].item()

print("CA velocities: {:.2f},   {:.2f}".format(v_2dca, w_2dca))
print("CC velocities: {:.2f},   {:.2f}".format(v_2dcc, w_2dcc))
print("CA max velocities: {:.2f},   {:.2f}".format(v_2dca_max, w_2dca_max))
print("CC max velocities: {:.2f},   {:.2f}".format(v_2dcc_max, w_2dcc_max))
print("Input velocities: {:.2f},   {:.2f}".format(v_input, w_input))


def plot_component(i, ax, position_ca, position_cc, mask_ca, mask_cc):
    input_velocity = v_input if i == 0 else w_input
    ax.scatter(position_ca.time, position_ca.values[:, i], color="blue", s=0.5)
    ax.scatter(position_cc.time, position_cc.values[:, i], color="green", s=0.5)
    ax.plot(
        position_ca.time,
        4 + position_ca.time.values * input_velocity,
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
    ylabel = r"$x(t)$" if i == 0 else r"$y(t)$"
    ax.set_ylabel(ylabel)


plot_component(0, axes[0, 0], pos_2dca, pos_2dca, mask_2dca, mask_2dca)
plot_component(1, axes[0, 1], pos_2dca, pos_2dca, mask_2dca, mask_2dca)
plot_component(0, axes[1, 0], pos_max_2dca, pos_max_2dcc, mask_max_2dca, mask_max_2dcc)
plot_component(1, axes[1, 1], pos_max_2dca, pos_max_2dcc, mask_max_2dca, mask_max_2dcc)

axes[0, 0].set_title("Contouring")
axes[0, 1].set_title("Contouring")

axes[1, 0].set_title("Max track.")
axes[1, 1].set_title("Max track.")

axes[1, 0].set_xlabel("Time")
axes[1, 1].set_xlabel("Time")

plt.savefig("barberpole_trajectories.pdf", bbox_inches="tight")

plt.show()

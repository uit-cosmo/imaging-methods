import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp
import xarray as xr

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)


mp = im.get_default_synthetic_method_params()

T = 5000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 5000

N = 5
NSR = 0.1

lx, ly = 0.5, 2.0
theta = 0
i = 0

file_name = os.path.join(
    "synthetic_data", "data_{:.2f}_{:.2f}_{:.2f}_{}".format(lx, ly, theta, i)
)
ds = xr.open_dataset(file_name)

dt = im.get_dt(ds)
dr = im.get_dr(ds)

tdca_params = mp.two_dca
events, average_ds = im.find_events_and_2dca(
    ds,
    tdca_params.refx,
    tdca_params.refy,
    threshold=tdca_params.threshold,
    check_max=tdca_params.check_max,
    window_size=tdca_params.window,
    single_counting=tdca_params.single_counting,
)


def get_positions_and_mask(
    average_ds, variable, method_parameters, position_method="contouring"
):
    if position_method == "contouring":
        position_da = im.get_contour_evolution(
            average_ds[variable],
            method_parameters.contouring.threshold_factor,
            max_displacement_threshold=None,
        ).center_of_mass
    elif position_method == "max":
        position_da = im.compute_maximum_trajectory_da(
            average_ds, variable, method="fit"
        )
    else:
        raise NotImplementedError

    position_da, start, end = im.smooth_da(
        position_da, method_parameters.position_smoothing, return_start_end=True
    )
    signal_high = (
        average_ds[variable].max(dim=["x", "y"]).values
        > 0.75 * average_ds[variable].max().item()
    )[start:end]
    mask = im.get_combined_mask(
        average_ds, position_da, signal_high, 2 * im.get_dr(average_ds)
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

fig, axes = plt.subplots(1, 2)

v_input, w_input = ds["v_input"].item(), ds["w_input"].item()

print("CA velocities: {:.2f},   {:.2f}".format(v_2dca, w_2dca))
print("CC velocities: {:.2f},   {:.2f}".format(v_2dcc, w_2dcc))
print("CA max velocities: {:.2f},   {:.2f}".format(v_2dca_max, w_2dca_max))
print("CC max velocities: {:.2f},   {:.2f}".format(v_2dcc_max, w_2dcc_max))
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


# plot_component(0, axes[0], contour_ca.center_of_mass, contour_cc.center_of_mass)
# plot_component(1, axes[1], contour_ca.center_of_mass, contour_cc.center_of_mass)
plot_component(0, axes[0], pos_max_2dca, pos_max_2dcc, mask_max_2dca, mask_max_2dcc)
plot_component(1, axes[1], pos_max_2dca, pos_max_2dcc, mask_max_2dca, mask_max_2dcc)

plt.show()

print("LOL")

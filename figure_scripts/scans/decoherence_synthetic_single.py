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


# Method parameters
method_parameters = {
    "preprocessing": {"radius": 1000},
    "2dca": {
        "refx": 8,
        "refy": 8,
        "threshold": 2,
        "window": 60,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.5, "com_smoothing": 10},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

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

lx, ly = 1, 1
theta = 0
i = 2
rand_coeff = 0.67

file_name = os.path.join(
    "synthetic_data", "decoherence_data_{:.2f}_{}".format(rand_coeff, i)
)
ds = xr.open_dataset(file_name)

dt = im.get_dt(ds)
dr = im.get_dr(ds)

tdca_params = method_parameters["2dca"]
events, average_ds = im.find_events_and_2dca(
    ds,
    tdca_params["refx"],
    tdca_params["refy"],
    threshold=tdca_params["threshold"],
    check_max=tdca_params["check_max"],
    window_size=tdca_params["window"],
    single_counting=tdca_params["single_counting"],
)


def get_contouring_velocities(variable):
    contour_ds = im.get_contour_evolution(
        average_ds[variable],
        method_parameters["contouring"]["threshold_factor"],
        max_displacement_threshold=None,
    )
    signal_high = (
        average_ds[variable].max(dim=["x", "y"]).values
        > 0.75 * average_ds[variable].max().item()
    )
    mask = im.get_combined_mask(
        average_ds, contour_ds.center_of_mass, signal_high, 2 * dr
    )

    v, w = im.get_averaged_velocity_from_position(
        position_da=contour_ds.center_of_mass, mask=mask, window_size=1
    )
    return v, w


def get_max_pos_velocities(variable):
    max_trajectory = im.compute_maximum_trajectory_da(
        average_ds, variable, method="fit"
    )
    signal_high = (
        average_ds[variable].max(dim=["x", "y"]).values
        > 0.75 * average_ds[variable].max().item()
    )
    mask = im.get_combined_mask(average_ds, max_trajectory, signal_high, 2 * dr)

    v, w = im.get_averaged_velocity_from_position(
        position_da=max_trajectory, mask=mask, window_size=1
    )
    return v, w


v_2dca, w_2dca = get_contouring_velocities("cond_av")
v_2dcc, w_2dcc = get_contouring_velocities("cross_corr")

contour_ca = im.get_contour_evolution(
    average_ds.cond_av,
    method_parameters["contouring"]["threshold_factor"],
    max_displacement_threshold=None,
)

contour_cc = im.get_contour_evolution(
    average_ds.cross_corr,
    method_parameters["contouring"]["threshold_factor"],
    max_displacement_threshold=None,
)

ca_max_mask = (
    average_ds.cond_av.max(dim=["x", "y"]).values
    > 0.75 * average_ds.cond_av.max().item()
)
cc_max_mask = (
    average_ds.cross_corr.max(dim=["x", "y"]).values
    > 0.75 * average_ds.cross_corr.max().item()
)

fig, axes = plt.subplots(1, 2)

v_input, w_input = ds["v_input"].item(), ds["w_input"].item()

print("CA velocities: {:.2f},   {:.2f}".format(v_2dca, w_2dca))
print("CC velocities: {:.2f},   {:.2f}".format(v_2dcc, w_2dcc))
print("Input velocities: {:.2f},   {:.2f}".format(v_input, w_input))
ax = axes[0]


def plot_component(i, ax):
    input_velocity = v_input if i == 0 else w_input
    ax.plot(contour_ca.time, contour_ca.center_of_mass.values[:, i], color="blue")
    ax.plot(contour_ca.time, contour_cc.center_of_mass.values[:, i], color="green")
    ax.plot(
        contour_ca.time,
        contour_ca.center_of_mass.sel(time=0).values[0]
        + contour_ca.time.values * input_velocity,
        color="black",
        ls="--",
    )

    ca_combined_mask = im.get_combined_mask(
        average_ds, contour_ca.center_of_mass, ca_max_mask, 2 * dr
    )
    cc_combined_mask = im.get_combined_mask(
        average_ds, contour_cc.center_of_mass, cc_max_mask, 2 * dr
    )
    ax.plot(
        contour_ca.time[ca_combined_mask],
        contour_ca.center_of_mass.values[:, i][ca_combined_mask],
        lw=2,
        color="blue",
    )
    ax.plot(
        contour_cc.time[cc_combined_mask],
        contour_cc.center_of_mass.values[:, i][cc_combined_mask],
        lw=2,
        color="green",
    )


plot_component(0, axes[0])
plot_component(1, axes[1])


plt.show()


print("LOL")

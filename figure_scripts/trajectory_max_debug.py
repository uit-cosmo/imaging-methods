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

lx, ly = 0.5, 2.0
theta = 0
i = 0

file_name = os.path.join(
    "synthetic_data", "data_{:.2f}_{:.2f}_{:.2f}_{}".format(lx, ly, theta, i)
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


def get_max_pos_velocities(variable):
    max_trajectory = im.compute_maximum_trajectory_da(
        average_ds, variable, method="fit"
    )
    signal_high = average_ds[variable].max(dim=["x", "y"]).values > 0.75
    mask = im.get_combined_mask(average_ds, max_trajectory, signal_high, dr)

    v, w = im.get_averaged_velocity_from_position(
        position_da=max_trajectory, mask=mask, window_size=1
    )
    return v, w


max_trajectory_ca = im.compute_maximum_trajectory_da(
    average_ds, "cond_av", method="parabolic"
)
max_trajectory_cc = im.compute_maximum_trajectory_da(
    average_ds, "cross_corr", method="parabolic"
)

fig, ax = plt.subplots(1, 2)

ax[0].plot(max_trajectory_ca.values[:, 0], max_trajectory_ca.values[:, 1], color="blue")
ax[0].plot(max_trajectory_cc.values[:, 0], max_trajectory_cc.values[:, 1], color="red")

ax[1].plot(max_trajectory_ca.time, max_trajectory_ca.values[:, 0], color="blue")
ax[1].plot(max_trajectory_cc.time, max_trajectory_cc.values[:, 0], color="red")

plt.savefig("trajectories.pdf", bbox_inches="tight")
plt.show()

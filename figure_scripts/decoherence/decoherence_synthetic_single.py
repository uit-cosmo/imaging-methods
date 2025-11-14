import numpy as np

from decoherence_utils import *
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

T = 10000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 10000

vx_input = 1
vy_intput = 0
lx_input = 1
ly_input = 1

rand_coeff = 1
force_redo = False

file_name = "decoherence_{}.nc".format(rand_coeff)
if os.path.exists(file_name):
    ds = xr.open_dataset(file_name)
else:
    ds = make_decoherence_realization(rand_coeff=rand_coeff)
    ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
    ds.to_netcdf(file_name)

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

contour_ds = im.get_contour_evolution(
    average_ds.cond_av,
    method_parameters["contouring"]["threshold_factor"],
    max_displacement_threshold=None,
)

velocity_ds = im.get_contour_velocity(
    contour_ds.center_of_mass,
    method_parameters["contouring"]["com_smoothing"],
)

vx_c, vy_c, vx_cc_tde, vy_cc_tde, confidence, vx_2dca_tde, vy_2dca_tde, cond_repr = (
    estimate_velocities(ds, method_parameters)
)

gif_file_name = "contours_decoherence_synthetic_{}.gif".format(rand_coeff)

plot_contours = False

if plot_contours:
    im.show_movie_with_contours(
        average_ds,
        contour_ds,
        apd_dataset=None,
        variable="cond_av",
        lims=(0, 3),
        gif_name=gif_file_name,
        interpolation="spline16",
        show=False,
    )

    im.show_movie_with_contours(
        average_ds,
        contour_ds,
        apd_dataset=None,
        variable="cond_repr",
        lims=(0, 1),
        gif_name="cond_repr_{}.gif".format(rand_coeff),
        interpolation="spline16",
        show=False,
    )

print("CC 3TDE velocities: {:.2f} {:.2f}".format(vx_cc_tde, vy_cc_tde))
print("2DCA 3TDE velocities: {:.2f} {:.2f}".format(vx_2dca_tde, vy_2dca_tde))
print("C velocities: {:.2f} {:.2f}".format(vx_c, vy_c))
print("Confidence: {:.2f}".format(confidence))
print("Cond. Repr. Index: {:.2f}".format(cond_repr))

tau_x_index = int(1 / vx_c / dt)
print(
    "Cond repr at neighbour: {:.2f}".format(
        np.max(average_ds.cond_repr.isel(x=5, y=4).values)
    )
)

# === Enhanced Plotting Section ===
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, constrained_layout=True)

time = average_ds.time.values

# --- Plot 1: Conditional Average (cond_av) at neighboring pixels ---
ax = axes[0]
label_data = [
    ("left", average_ds.cond_av.isel(x=3, y=4)),
    ("center", average_ds.cond_av.isel(x=4, y=4)),
    ("right", average_ds.cond_av.isel(x=5, y=4)),
    ("down", average_ds.cond_av.isel(x=4, y=3)),
    ("up", average_ds.cond_av.isel(x=4, y=5)),
]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(label_data)))

for (label, data), color in zip(label_data, colors):
    ax.plot(time, data, label=label, color=color, lw=1.5)

ax.set_ylabel(r"$\langle n \rangle$ (cond. avg.)")
ax.set_title("Conditional Average at Neighboring Pixels")
ax.legend(fontsize=9, loc="upper right", frameon=True, fancybox=False, edgecolor="k")
ax.grid(True, which="both", ls="--", alpha=0.5)

# --- Plot 2: Conditional Representation (cond_repr) ---

label_data = [
    ("left", average_ds.cond_repr.isel(x=3, y=4)),
    ("center", average_ds.cond_repr.isel(x=4, y=4)),
    ("right", average_ds.cond_repr.isel(x=5, y=4)),
    ("down", average_ds.cond_repr.isel(x=4, y=3)),
    ("up", average_ds.cond_repr.isel(x=4, y=5)),
]
ax = axes[1]
for (label, data), color in zip(label_data, colors):
    ax.plot(time, data.values, color=color, lw=1.5)

ax.set_ylabel("Cond. Repr.")
ax.set_title("Conditional Reproducibility at Neighboring Pixels")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, which="both", ls="--", alpha=0.5)

# --- Plot 3: Global Maxima over Space ---
ax = axes[2]
ax.plot(
    time,
    average_ds.cond_av.max(dim=["x", "y"]) / average_ds.cond_av.max(),
    label="Max of cond. avg.",
    color="tab:blue",
    lw=2,
)
ax.plot(
    time,
    average_ds.cond_repr.max(dim=["x", "y"]),
    label="Max of cond. repr.",
    color="tab:orange",
    lw=2,
)

ax.set_xlabel("Time")
ax.set_ylabel("Spatial Maximum")
ax.set_title("Global Spatial Maxima Over Time")
ax.legend(fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.5)

# Final adjustments
fig.suptitle(
    "Decoherence Analysis: Conditional Statistics", fontsize=14, fontweight="bold"
)
plt.savefig(
    "decoherence_analysis_random_{}.pdf".format(rand_coeff), bbox_inches="tight"
)
plt.show()

import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp
import numpy as np
import xarray as xr

from imaging_methods import movie_dataset
from scan_utils import *

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

method_parameters = im.get_default_synthetic_method_params()

data_file = "cc_data.npz"
data = "data_cc"

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

def get_simulation_data(a, i):
    file_name = os.path.join(
        "synthetic_data", "{}_{}_{}".format(data, a, i)
    )
    if os.path.exists(file_name):
        return xr.open_dataset(file_name)
    alpha = 0 #  np.random.uniform(-np.pi / 4, np.pi / 4)
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
        lx=1,
        ly=1,
        theta=0,
        bs=bs,
    )
    ds_mean = ds.frames.mean().item()
    ds = ds.assign(
        frames=ds["frames"] + ds_mean * NSR * np.random.random(ds.frames.shape)
    )

    amplitude = a  # Wave amplitude
    wavelength = 1.0  # Wavelength in x units (e.g., meters)
    period = 2.0  # Period in time units (e.g., seconds)
    wave_speed = wavelength / period  # Speed = lambda / T

    # Step 3: Create the propagating wave DataArray
    # Simple 1D wave propagating in +x direction: A * sin(2π (x/λ - t/T))
    # Use xarray broadcasting: expand to match (time, x, y)
    k = 2 * np.pi / wavelength  # Wave number
    omega = 2 * np.pi / period  # Angular frequency
    wave = amplitude * np.sin(k * ds.y - omega * ds.time)  # Shape: (time, x)
    wave = wave.expand_dims(x=ds.x)  # Broadcast to (time, x, y)

    # Step 4: Add the wave to the original dataset (creates a new variable)
    ds['frames'] = ds['frames'] + wave

    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    ds["v_input"] = vx_input
    ds["w_input"] = vy_input
    ds.to_netcdf(file_name)
    return ds


amplitudes = np.linspace(0, 2, num=10)  # change num= back to 3 if you wish

# Containers for *all* realisations (list of lists)
v_2dca_all = []  # len = len(thetas); each entry = list of N values
w_2dca_all = []
v_2dcc_all = []  # len = len(thetas); each entry = list of N values
w_2dcc_all = []
v_2dca_max_all = []
w_2dca_max_all = []
v_2dcc_max_all = []
w_2dcc_max_all = []
v_tde_all = []
w_tde_all = []

if os.path.exists(data_file):
    loaded = np.load(data_file)
    amplitudes = loaded["amplitudes"]
    v_2dca_all = loaded["v_2dca_all"]
    w_2dca_all = loaded["w_2dca_all"]
    v_2dcc_all = loaded["v_2dcc_all"]
    w_2dcc_all = loaded["w_2dcc_all"]
    v_2dca_max_all = loaded["v_2dca_max_all"]
    w_2dca_max_all = loaded["w_2dca_max_all"]
    v_2dcc_max_all = loaded["v_2dcc_max_all"]
    w_2dcc_max_all = loaded["w_2dcc_max_all"]
    v_tde_all = loaded["vxtde_all"]
    w_tde_all = loaded["vytde_all"]
else:
    for a in amplitudes:
        print(f"Processing amplitude = {a:.3f}")
        (
            v_2dca,
            w_2dca,
            v_2dcc,
            w_2dcc,
            v_2dca_max,
            w_2dca_max,
            v_2dcc_max,
            w_2dcc_max,
            vxtde,
            vytde,
        ) = get_all_velocities(
            N, lambda i: get_simulation_data(a, i), method_parameters
        )

        v_2dca_all.append(v_2dca)
        w_2dca_all.append(w_2dca)
        v_2dcc_all.append(v_2dcc)
        w_2dcc_all.append(w_2dcc)
        v_2dca_max_all.append(v_2dca_max)
        w_2dca_max_all.append(w_2dca_max)
        v_2dcc_max_all.append(v_2dcc_max)
        w_2dcc_max_all.append(w_2dcc_max)
        v_tde_all.append(vxtde)
        w_tde_all.append(vytde)

    np.savez(
        data_file,
        amplitudes=amplitudes,
        v_2dca_all=v_2dca_all,
        w_2dca_all=w_2dca_all,
        v_2dcc_all=v_2dcc_all,
        w_2dcc_all=w_2dcc_all,
        v_2dca_max_all=v_2dca_max_all,
        w_2dca_max_all=w_2dca_max_all,
        v_2dcc_max_all=v_2dcc_max_all,
        w_2dcc_max_all=w_2dcc_max_all,
        vxtde_all=v_tde_all,
        vytde_all=w_tde_all,
    )


v_2dca_all = np.array(v_2dca_all)
w_2dca_all = np.array(w_2dca_all)
v_2dcc_all = np.array(v_2dcc_all)
w_2dcc_all = np.array(w_2dcc_all)
v_2dca_max_all = np.array(v_2dca_max_all)
w_2dca_max_all = np.array(w_2dca_max_all)
v_2dcc_max_all = np.array(v_2dcc_max_all)
w_2dcc_max_all = np.array(w_2dcc_max_all)
v_tde_all = np.array(v_tde_all)
w_tde_all = np.array(w_tde_all)

# --------------------------------------------------------------
# 3.  SCATTER PLOT
# --------------------------------------------------------------
fig, ax = plt.subplots()


# Helper: scatter one component
def scatter_component(theta_vals, data_per_theta, label, marker, color):
    first = True
    for th, vals in zip(theta_vals, data_per_theta):
        jitter = 0  # np.random.normal(0, 0.003, size=len(vals))
        ax.scatter(
            np.full_like(vals, th) + jitter,
            vals,
            marker=marker,
            s=10,
            edgecolor="k",
            linewidth=0.3,
            label=label if first else None,  # <-- only label first
            alpha=0.7,
            color=color,
            )
        first = False


# Choose distinct colours (you can also use a colormap)
labelc = r"Contouring"
labelmax = r"Max. Track."
labeltde = r"Time delay est."

scatter_component(
        amplitudes, v_2dca_all**2 + w_2dca_all**2, labelc, "o", "#1f77b4"
    )
scatter_component(
        amplitudes, v_2dcc_all**2 + w_2dcc_all**2, labelc, "o", "#1f77b4"
    )

ax.set_xticks([0, 1 / 4, 1 / 2])
ax.set_xticklabels([r"$0$", r"$1/4$", r"$1/2$"])
ax.set_xlabel(r"$\alpha/\pi$")
ax.set_ylabel("Error")
ax.legend()  # loc=6
ax.set_ylim(-0.1, 1.1)

plt.savefig("cc.eps", bbox_inches="tight")
plt.show()

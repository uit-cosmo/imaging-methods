import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import os
import cosmoplots as cp
import numpy as np
import xarray as xr
from scan_utils import *

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
        "refx": 4,
        "refy": 4,
        "threshold": 2,
        "window": 60,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 10},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

data_file = "size_scan_data.npz"
force_redo = False

T = 5000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 5000
NSR = 0.1

vx_input = 1
vy_intput = 0
lx_input = 1
ly_input = 1
N = 1


def get_simulation_data(l, i):
    file_name = os.path.join("synthetic_data", "data_size_{:.2f}_{}".format(l, i))
    if os.path.exists(file_name):
        return xr.open_dataset(file_name)

    theta = np.random.uniform(-np.pi / 4, np.pi / 4)
    vx_input, vy_input = np.cos(theta), np.sin(theta)
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
        lx=l,
        ly=l,
        theta=0,
        bs=bs,
        periodic_y=False,  # Use only for w=0 so periodicity doesn't matter, for large blobs, periodicity has issues.
    )
    ds_mean = ds.frames.mean().item()
    ds = ds.assign(
        frames=ds["frames"] + ds_mean * NSR * np.random.random(ds.frames.shape)
    )
    ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
    ds["v_input"] = vx_input
    ds["w_input"] = vy_input
    ds.to_netcdf(file_name)
    return ds


sizes = np.logspace(-1, np.log10(4), num=10)
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


if os.path.exists(data_file) and not force_redo:
    loaded = np.load(data_file)
    sizes = loaded["sizes"]
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
    for l in sizes:
        print(f"Processing l = {l:.3f}")
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
            N, lambda i: get_simulation_data(l, i), method_parameters
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
        sizes=sizes,
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
            s=40,
            edgecolor="k",
            linewidth=0.3,
            label=label if first else None,  # <-- only label first
            alpha=0.7,
            color=color,
        )
        first = False


# Choose distinct colours (you can also use a colormap)
# labelc = r"$\sqrt{(v_c-v)^2+(w_c-w)^2}$"
# labeltde = r"$\sqrt{(v_{\mathrm{TDE}}-v)^2+(w_{\mathrm{TDE}}-w)^2}$"
labelc = r"Contouring"
labelmax = r"Max. Track."
labeltde = r"Time delay est."

scatter_component(1 / sizes, v_2dca_all**2 + w_2dca_all**2, labelc, "o", "#1f77b4")
scatter_component(
    1 / sizes, v_2dca_max_all**2 + w_2dca_max_all**2, labelmax, "s", "red"
)
scatter_component(1 / sizes, v_tde_all**2 + w_tde_all**2, labeltde, "D", "#2ca02c")

ax.set_xlabel(r"$\Delta/\ell$")
ax.set_ylabel("Error")
ax.legend()  # loc=6
ax.set_xscale("log")
ax.set_xticks([1 / 4, 1, 10])
ax.set_xticklabels([r"$1/4$", r"$1$", r"$10$"])

ax.set_ylim(-0.2, 1.2)

plt.savefig("size_scan.eps", bbox_inches="tight")
plt.show()

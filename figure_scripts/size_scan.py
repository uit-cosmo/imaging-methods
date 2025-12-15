from utils import *
import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp

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
force_redo = True

i = 0


def estimate_velocities(ds, method_parameters):
    """
    Does a full analysis on imaging data, estimating a BlobParameters object containing all estimates. Plots figures
    when relevant in the provided figures_dir.
    """
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
        signal_high = average_ds[variable].max(dim=["x", "y"]).values > 0.75
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
        signal_high = average_ds[variable].max(dim=["x", "y"]).values > 0.75
        mask = im.get_combined_mask(average_ds, max_trajectory, signal_high, dr)

        v, w = im.get_averaged_velocity_from_position(
            position_da=max_trajectory, mask=mask, window_size=1
        )
        return v, w

    v_2dca, w_2dca = get_contouring_velocities("cond_av")
    v_2dcc, w_2dcc = get_contouring_velocities("cross_corr")

    v_2dca_max, w_2dca_max = get_max_pos_velocities("cond_av")
    v_2dcc_max, w_2dcc_max = get_max_pos_velocities("cross_corr")

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters["2dca"]["window"] * dt
    eo.cc_options.minimum_cc_value = 0
    pd = ve.estimate_velocities_for_pixel(
        tdca_params["refx"], tdca_params["refy"], ve.CModImagingDataInterface(ds)
    )
    vx, vy = pd.vx, pd.vy

    return (
        v_2dca,
        w_2dca,
        v_2dcc,
        w_2dcc,
        v_2dca_max,
        w_2dca_max,
        v_2dcc_max,
        w_2dcc_max,
        vx,
        vy,
    )


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
    ds = make_2d_realization(
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


def get_all_velocities(l, N=N):
    """
    Run N realisations and return the *raw* velocity components.

    Returns
    -------
    vx_all, vy_all, vxtde_all, vytde_all : list (length N)
        One entry per Monte-Carlo realisation.
    """
    v_2dca_all = []
    w_2dca_all = []
    v_2dcc_all = []
    w_2dcc_all = []
    v_2dca_max_all = []
    w_2dca_max_all = []
    v_2dcc_max_all = []
    w_2dcc_max_all = []
    vxtde_all = []
    vytde_all = []

    for i in range(N):
        ds = get_simulation_data(l, i)
        v_input = ds["v_input"]
        w_input = ds["w_input"]

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
        ) = estimate_velocities(ds, method_parameters)

        v_2dca_all.append(v_2dca - v_input)
        w_2dca_all.append(w_2dca - w_input)
        v_2dcc_all.append(v_2dcc - v_input)
        w_2dcc_all.append(w_2dcc - w_input)
        v_2dca_max_all.append(v_2dca_max - v_input)
        w_2dca_max_all.append(w_2dca_max - w_input)
        v_2dcc_max_all.append(v_2dcc_max - v_input)
        w_2dcc_max_all.append(w_2dcc_max - w_input)
        vxtde_all.append(vxtde - v_input)
        vytde_all.append(vytde - w_input)

    return (
        v_2dca_all,
        w_2dca_all,
        v_2dcc_all,
        w_2dcc_all,
        v_2dca_max_all,
        w_2dca_max_all,
        v_2dcc_max_all,
        w_2dcc_max_all,
        vxtde_all,
        vytde_all,
    )


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
    thetas = loaded["thetas"]  # in case you want to load thetas too
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
        ) = get_all_velocities(l, N=N)

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
labelc = r"$E_c$"
labeltde = r"$E_{\mathrm{TDE}}$"
scatter_component(1 / sizes, np.sqrt(vx_all**2 + vy_all**2), labelc, "o", "#1f77b4")
scatter_component(
    1 / sizes, np.sqrt(vxtde_all**2 + vytde_all**2), labeltde, "^", "#2ca02c"
)

ax.set_xlabel(r"$\Delta/\ell$")
ax.set_ylabel("Error")
ax.legend()  # loc=6
ax.set_xscale("log")
ax.set_xticks([1 / 4, 1, 10])
ax.set_xticklabels([r"$1/4$", r"$1$", r"$10$"])

plt.savefig("size_scan.eps", bbox_inches="tight")
plt.show()

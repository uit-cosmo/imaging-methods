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

data_file = "barberpole_data.npz"
force_redo = True


def estimate_velocities(ds, method_parameters):
    """
    Does a full analysis on imaging data, estimating a BlobParameters object containing all estimates. Plots figures
    when relevant in the provided figures_dir.
    """
    dt = im.get_dt(ds)

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
        velocity_ds = im.get_contour_velocity(
            contour_ds.center_of_mass,
            method_parameters["contouring"]["com_smoothing"],
        )
        v_c, w_c = im.get_average_velocity_for_near_com(
            average_ds, contour_ds, velocity_ds, distance=1
        )
        return v_c, w_c

    v_2dca, w_2dca = get_contouring_velocities("cond_av")
    v_2dcc, w_2dcc = get_contouring_velocities("cross_corr")

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters["2dca"]["window"] * dt
    eo.cc_options.minimum_cc_value = 0
    pd = ve.estimate_velocities_for_pixel(
        tdca_params["refx"], tdca_params["refy"], ve.CModImagingDataInterface(ds)
    )
    vx, vy = pd.vx, pd.vy

    return v_2dca, w_2dca, v_2dcc, w_2dcc, vx, vy


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


def get_simulation_data(lx, ly, theta, i):
    file_name = os.path.join(
        "synthetic_data", "data_{:.2f}_{:.2f}_{:.2f}_{}".format(lx, ly, theta, i)
    )
    if os.path.exists(file_name):
        return xr.open_dataset(file_name)
    alpha = np.random.uniform(-np.pi / 4, np.pi / 4)
    vx_input, vy_input = np.cos(alpha), np.sin(alpha)
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
        lx=lx,
        ly=ly,
        theta=theta + alpha,
        bs=bs,
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


def get_all_velocities(lx, ly, theta, N=N):
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
    vxtde_all = []
    vytde_all = []

    for i in range(N):
        ds = get_simulation_data(lx, ly, theta, i)
        v_input = ds["v_input"]
        w_input = ds["w_input"]

        v_2dca, w_2dca, v_2dcc, w_2dcc, vxtde, vytde = estimate_velocities(
            ds, method_parameters
        )

        v_2dca_all.append(v_2dca - v_input)
        w_2dca_all.append(w_2dca - w_input)
        v_2dcc_all.append(v_2dcc - v_input)
        w_2dcc_all.append(w_2dcc - w_input)
        vxtde_all.append(vxtde - v_input)
        vytde_all.append(vytde - w_input)

    return v_2dca_all, w_2dca_all, v_2dcc_all, w_2dcc_all, vxtde_all, vytde_all


# --------------------------------------------------------------
# 2.  SWEEP OVER theta
# --------------------------------------------------------------
lx, ly = 0.5, 2
thetas = np.linspace(0, np.pi / 2, num=20)  # change num= back to 3 if you wish

# Containers for *all* realisations (list of lists)
v_2dca_all = []  # len = len(thetas); each entry = list of N values
w_2dca_all = []
v_2dcc_all = []  # len = len(thetas); each entry = list of N values
w_2dcc_all = []
v_tde_all = []
w_tde_all = []

if os.path.exists(data_file) and not force_redo:
    loaded = np.load(data_file)
    thetas = loaded["thetas"]  # in case you want to load thetas too
    v_2dca_all = loaded["v_2dca_all"]
    w_2dca_all = loaded["w_2dca_all"]
    v_2dcc_all = loaded["v_2dcc_all"]
    w_2dcc_all = loaded["w_2dcc_all"]
    v_tde_all = loaded["vxtde_all"]
    w_tde_all = loaded["vytde_all"]
else:
    for theta in thetas:
        print(f"Processing theta = {theta:.3f} rad ({np.degrees(theta):.1f}°)")
        v_2dca, w_2dca, v_2dcc, w_2dcc, vxtde, vytde = get_all_velocities(
            lx, ly, theta, N=N
        )

        v_2dca_all.append(v_2dca)
        w_2dca_all.append(w_2dca)
        v_2dcc_all.append(v_2dcc)
        w_2dcc_all.append(w_2dcc)
        v_tde_all.append(vxtde)
        w_tde_all.append(vytde)

    np.savez(
        data_file,
        thetas=thetas,
        v_2dca_all=v_2dca_all,
        w_2dca_all=w_2dca_all,
        v_2dcc_all=v_2dcc_all,
        w_2dcc_all=w_2dcc_all,
        vxtde_all=v_tde_all,
        vytde_all=w_tde_all,
    )

v_2dca_all = np.array(v_2dca_all)
w_2dca_all = np.array(w_2dca_all)
v_2dcc_all = np.array(v_2dcc_all)
w_2dcc_all = np.array(w_2dcc_all)
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


def taumax(dx, dy, lpara, lperp, v, w, a):
    d1 = (dx * lperp**2 * v + dy * lpara**2 * w) * np.cos(a) ** 2
    d2 = (dx * lpara**2 * v + dy * lperp**2 * w) * np.sin(a) ** 2
    d3 = -(lpara**2 - lperp**2) * (dy * v + dx * w) * np.cos(a) * np.sin(a)
    n1 = (lperp**2 * v**2 + lpara**2 * w**2) * np.cos(a) ** 2
    n2 = -2 * (lpara**2 - lperp**2) * v * w * np.sin(a) * np.cos(a)
    n3 = (lpara**2 * v**2 + lperp**2 * w**2) * np.sin(a) ** 2
    return (d1 + d2 + d3) / (n1 + n2 + n3)


def v3(lpara, lperp, v, w, a):
    tx = taumax(1, 0, lpara, lperp, v, w, a)
    ty = taumax(0, 1, lpara, lperp, v, w, a)
    return tx / (tx**2 + ty**2)


def w3(lpara, lperp, v, w, a):
    tx = taumax(1, 0, lpara, lperp, v, w, a)
    ty = taumax(0, 1, lpara, lperp, v, w, a)
    return ty / (tx**2 + ty**2)


# Choose distinct colours (you can also use a colormap)
# labelc = r"$\sqrt{(v_c-v)^2+(w_c-w)^2}$"
# labeltde = r"$\sqrt{(v_{\mathrm{TDE}}-v)^2+(w_{\mathrm{TDE}}-w)^2}$"
labelc = r"$E_c$"
labeltde = r"$E_{\mathrm{TDE}}$"
scatter_component(
    thetas / np.pi, np.sqrt(v_2dca_all**2 + w_2dca_all**2), labelc, "o", "#1f77b4"
)
scatter_component(
    thetas / np.pi, np.sqrt(v_2dcc_all**2 + w_2dcc_all**2), labelc, "s", "red"
)
scatter_component(
    thetas / np.pi, np.sqrt(v_tde_all**2 + w_tde_all**2), labeltde, "^", "#2ca02c"
)

v3s = np.array([v3(lx, ly, 1, 0, t) for t in thetas])
w3s = np.array([w3(lx, ly, 1, 0, t) for t in thetas])

ax.plot(thetas / np.pi, np.sqrt((v3s - 1) ** 2 + w3s**2), color="#2ca02c", ls="--")

ax.set_xticks([0, 1 / 4, 1 / 2])
ax.set_xticklabels([r"$0$", r"$1/4$", r"$1/2$"])
ax.set_xlabel(r"$\alpha/\pi$")
ax.set_ylabel("Error")
ax.legend()  # loc=6
ax.set_ylim(-0.1, 1.1)

plt.savefig("barberpole.eps", bbox_inches="tight")
plt.show()

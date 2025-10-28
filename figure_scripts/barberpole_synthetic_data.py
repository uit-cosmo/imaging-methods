from utils import *
import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

# Method parameters
method_parameters = {
    "preprocessing": {"radius": 1000},
    "2dca": {
        "refx": 4,
        "refy": 4,
        "threshold": 2,
        "window": 30,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 3},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

i=0
def estimate_velocities(
    ds, method_parameters
):
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

    contour_ds = im.get_contour_evolution(
        average_ds.cond_av,
        method_parameters["contouring"]["threshold_factor"],
        max_displacement_threshold=None,
    )

    velocity_ds = im.get_contour_velocity(
        contour_ds.center_of_mass,
        method_parameters["contouring"]["com_smoothing"],
    )
    do_plots = False

    if do_plots:
        global i
        gif_file_name = "contours_{}.gif".format(i)
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
        i += 1

    v_c, w_c = velocity_ds.sel(time=slice(-3, 3)).mean(dim="time", skipna=True).values

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters["2dca"]["window"] * dt
    eo.cc_options.minimum_cc_value = 0.2
    pd = ve.estimate_velocities_for_pixel(tdca_params["refx"], tdca_params["refy"], ve.CModImagingDataInterface(ds))
    vx, vy = pd.vx, pd.vy

    return v_c, w_c, vx, vy


T = 1000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 1000

vx_input = 1
vy_intput = 0
lx_input = 1/2
ly_input = 2
theta_input = - np.pi/4


def get_velocities_for_parameters(lx, ly, theta):
    vx_list = []
    vy_list = []
    vxtde_list = []
    vytde_list = []

    N=1
    for _ in range(N):
        # ---- one realisation -------------------------------------------------
        ds = make_2d_realization(
            Lx, Ly, T,
            nx, ny, dt, K,
            vx=vx_input,
            vy=vy_intput,
            lx=lx, ly=ly,
            theta=theta,
            bs=bs,
        )
        ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
        vx, vy, vxtde, vytde = estimate_velocities(ds, method_parameters)

        vx_list.append(vx)
        vy_list.append(vy)
        vxtde_list.append(vxtde)
        vytde_list.append(vytde)

    return (
        np.mean(vx_list),
        np.mean(vy_list),
        np.mean(vxtde_list),
        np.mean(vytde_list),
    )

lx, ly = 1/2, 2
thetas = np.linspace(0, np.pi/2, num=3)

vx_avgs, vy_avgs, vxtde_avgs, vytde_avgs = [], [], [], []

for theta in thetas:
    print(f"Processing theta = {theta:.3f} rad ({theta*180/np.pi:.1f}°)")

    vx, vy, vxtde, vytde = get_velocities_for_parameters(lx, ly, theta)
    vx_avgs.append(vx)
    vy_avgs.append(vy)
    vxtde_avgs.append(vxtde)
    vytde_avgs.append(vytde)

# Convert to arrays
vx_avgs = np.array(vx_avgs)
vy_avgs = np.array(vy_avgs)
vxtde_avgs = np.array(vxtde_avgs)
vytde_avgs = np.array(vytde_avgs)

fig, ax = plt.subplots()

ax.plot(thetas, vx_avgs,     label=r'$v_x$',     marker='o')
ax.plot(thetas, vy_avgs,     label=r'$w_y$',     marker='s')
ax.plot(thetas, vxtde_avgs, label=r'$v_{\text{TDE}}$', marker='^')
ax.plot(thetas, vytde_avgs, label=r'$w_{\text{TDE}}$', marker='d')

ax.set_xlabel(r'$\theta$ (radians)')
ax.set_ylabel('Velocity estimate (averaged over N realizations)')
ax.legend()

plt.show()

from test_utils import *
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import cosmoplots as cp
from imaging_methods import *

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)


MAKE_PLOTS = True
T = 5000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 5000

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
    "contouring": {
        "threshold_factor": 0.3,
        "threshold_factor_cc": 0.5,
        "com_smoothing": 10,
    },
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

figures_dir = "integrated_tests_figures"


def make_decoherence_realization(rand_coeff):
    def blob_getter():
        return get_blob(
            amplitude=np.random.exponential(),
            vx=np.random.uniform(1 - rand_coeff, 1 + rand_coeff),
            vy=np.random.uniform(-rand_coeff, rand_coeff),
            posx=np.random.uniform(0, Lx),
            posy=np.random.uniform(0, Ly),
            lx=1,
            ly=1,
            t_init=np.random.uniform(0, T),
            bs=bs,
            theta=0,  # 0 but go to the != 0 branch in the blob.py function
        )

    return make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=0,  # velocities are overriden
        vy=0,
        lx=1,
        ly=1,
        theta=0,
        bs=bs,
        blob_getter=blob_getter,
    )


def movie(
    dataset: xr.Dataset,
    interval: int = 100,
    interpolation: str = "spline16",
    file_name=None,
) -> None:
    dt = get_dt(dataset)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    t_dim = "time"
    refx, refy = int(dataset["refx"].item()), int(dataset["refy"].item())

    contour_cc = get_contour_evolution(
        dataset.cross_corr,
        method_parameters["contouring"]["threshold_factor_cc"],
    )
    contour_ca = get_contour_evolution(
        dataset.cond_av,
        method_parameters["contouring"]["threshold_factor"],
    )
    max_ca = im.compute_maximum_trajectory_da(dataset, "cond_av")
    max_cc = im.compute_maximum_trajectory_da(dataset, "cross_corr")

    def get_title(i):
        time = dataset[t_dim][i]
        if dt < 1e-3:
            title = r"t$={:.2f}\,\mu$s".format(time * 1e6)
        else:
            title = r"t$={:.2f}\,$s".format(time)
        return title

    def animate_2d(i: int):
        arr0 = dataset.cond_av.isel(**{t_dim: i})
        arr1 = dataset.cross_corr.isel(**{t_dim: i})
        im0.set_data(arr0)
        im0.set_clim(0, np.max(dataset.cond_av.values))
        im1.set_data(arr1)
        im1.set_clim(0, np.max(dataset.cross_corr.values))

        c0 = contour_ca.contours.isel(time=i).data
        c1 = contour_cc.contours.isel(time=i).data
        line0[0].set_data(c0[:, 0], c0[:, 1])
        line1[0].set_data(c1[:, 0], c1[:, 1])
        com_scatter0.set_offsets(contour_ca.center_of_mass.values[i, :])
        com_scatter1.set_offsets(contour_cc.center_of_mass.values[i, :])
        max_scatter0.set_offsets(max_ca.values[i, :])
        max_scatter1.set_offsets(max_cc.values[i, :])

        tx.set_text(get_title(i))

    # tx = ax[0].set_title(get_title(0))
    tx = fig.suptitle(get_title(0))
    ax[0].set_title("Cond. Av.")
    ax[1].set_title("Cross Corr.")
    ax[0].scatter(
        dataset.R.isel(x=refx, y=refy).item(),
        dataset.Z.isel(x=refx, y=refy).item(),
        color="black",
    )
    ax[1].scatter(
        dataset.R.isel(x=refx, y=refy).item(),
        dataset.Z.isel(x=refx, y=refy).item(),
        color="black",
    )
    div0 = make_axes_locatable(ax[0])
    div1 = make_axes_locatable(ax[1])
    cax0 = div0.append_axes("right", "5%", "5%")
    cax1 = div1.append_axes("right", "5%", "5%")
    im0 = ax[0].imshow(
        dataset.cond_av.isel(**{t_dim: 0}),
        origin="lower",
        interpolation=interpolation,
    )
    im1 = ax[1].imshow(
        dataset.cross_corr.isel(**{t_dim: 0}),
        origin="lower",
        interpolation=interpolation,
    )
    com_scatter0 = ax[0].scatter(
        contour_ca.center_of_mass.values[0, 0],
        contour_ca.center_of_mass.values[0, 1],
        marker="s",
        color="green",
    )
    com_scatter1 = ax[1].scatter(
        contour_cc.center_of_mass.values[0, 0],
        contour_cc.center_of_mass.values[0, 1],
        marker="s",
        color="green",
    )
    max_scatter0 = ax[0].scatter(
        max_ca.values[0, 0], max_ca.values[0, 1], marker="^", color="orange"
    )
    max_scatter1 = ax[1].scatter(
        max_cc.values[0, 0], max_cc.values[0, 1], marker="^", color="orange"
    )
    line0 = ax[0].plot([], [], ls="--", color="black")
    line1 = ax[1].plot([], [], ls="--", color="black")
    fig.colorbar(im0, cax=cax0)
    fig.colorbar(im1, cax=cax1)

    im0.set_extent(
        (dataset.R[0, 0], dataset.R[0, -1], dataset.Z[0, 0], dataset.Z[-1, 0])
    )
    im1.set_extent(
        (dataset.R[0, 0], dataset.R[0, -1], dataset.Z[0, 0], dataset.Z[-1, 0])
    )

    ani = animation.FuncAnimation(
        fig, animate_2d, frames=dataset[t_dim].values.size, interval=interval
    )

    if file_name is not None:
        ani.save(file_name, writer="ffmpeg", fps=10)
        os.system(
            "gifsicle -i {} -O3 --colors 32 --lossy=150 -o {}".format(
                file_name, file_name
            )
        )

    plt.show()


def get_synthetic_data():
    alpha = np.pi / 8
    vx_input = np.cos(alpha)
    vy_intput = np.sin(alpha)
    lx_input = 1 / 2
    ly_input = 2
    theta_input = np.pi / 4

    ds = make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
    return ds


def test_real_data():
    shot = 1160616027
    manager = im.GPIDataAccessor(
        "/home/sosno/Git/experimental_database/plasma_discharges.json"
    )
    # ds = manager.read_shot_data(shot, preprocessed=True, data_folder="../data")
    ds = get_synthetic_data()

    tdca_params = method_parameters.two_dca
    events, average_ds = im.find_events_and_2dca(ds, tdca_params)
    movie(average_ds)

    method = "fit"
    cond_av_max = im.compute_maximum_trajectory_da(average_ds, "cond_av", method=method)
    cross_corr_max = im.compute_maximum_trajectory_da(
        average_ds, "cross_corr", method=method
    )
    refx, refy = tdca_params["refx"], tdca_params["refy"]
    delta = (
        average_ds.R.isel(x=refx, y=refy).item()
        - average_ds.R.isel(x=refx - 1, y=refy).item()
    )
    contour_ca = im.get_contour_evolution(
        average_ds.cond_av,
        method_parameters["contouring"]["threshold_factor"],
        max_displacement_threshold=None,
        com_method="centroid",
    )
    ca_centroid = contour_ca.center_of_mass
    contour_cc = im.get_contour_evolution(
        average_ds.cross_corr,
        method_parameters["contouring"]["threshold_factor_cc"],
        max_displacement_threshold=None,
        com_method="centroid",
    )
    ca_max = average_ds.cond_av.max(dim=["x", "y"]).values
    cc_max = average_ds.cross_corr.max(dim=["x", "y"]).values

    cc_centroid = contour_cc.center_of_mass

    fig, ax = plt.subplots()

    R, Z = (
        ds.R.isel(x=tdca_params["refx"], y=tdca_params["refy"]).item(),
        ds.Z.isel(x=tdca_params["refx"], y=tdca_params["refy"]).item(),
    )

    ax.scatter(R, Z)

    is_ca_high_enough = ca_max > 0.75 * np.max(ca_max)
    is_cc_high_enough = cc_max > 0.75 * np.max(cc_max)

    mask_ca_centroid = get_combined_mask(
        average_ds, ca_centroid, is_ca_high_enough, delta
    )
    mask_ca_max = get_combined_mask(average_ds, cond_av_max, is_ca_high_enough, delta)
    mask_cc_centroid = get_combined_mask(
        average_ds, cc_centroid, is_cc_high_enough, delta
    )
    mask_cc_max = get_combined_mask(
        average_ds, cross_corr_max, is_cc_high_enough, delta
    )

    ax.plot(ca_centroid.values[:, 0], ca_centroid.values[:, 1], color="blue", lw=0.5)
    ax.plot(
        ca_centroid.values[:, 0][mask_ca_centroid],
        ca_centroid.values[:, 1][mask_ca_centroid],
        color="blue",
        lw=1,
    )

    ax.plot(
        cond_av_max.values[:, 0],
        cond_av_max.values[:, 1],
        color="blue",
        ls="--",
        lw=0.5,
    )
    ax.plot(
        cond_av_max.values[:, 0][mask_ca_max],
        cond_av_max.values[:, 1][mask_ca_max],
        color="blue",
        ls="--",
        lw=1,
    )

    ax.plot(cc_centroid.values[:, 0], cc_centroid.values[:, 1], color="red", lw=0.5)

    ax.plot(
        cc_centroid.values[:, 0][mask_cc_centroid],
        cc_centroid.values[:, 1][mask_cc_centroid],
        color="red",
        lw=1,
    )

    ax.plot(
        cross_corr_max.values[:, 0],
        cross_corr_max.values[:, 1],
        color="red",
        ls="--",
        lw=0.5,
    )
    ax.plot(
        cross_corr_max.values[:, 0][mask_cc_max],
        cross_corr_max.values[:, 1][mask_cc_max],
        color="red",
        ls="--",
        lw=1,
    )

    ax.set_aspect("equal", adjustable="datalim")
    plt.savefig("trajectories.pdf", bbox_inches="tight")
    plt.show()

    v_ca_centroid, w_ca_centroid = get_averaged_velocity_from_position(
        ca_centroid, mask_ca_centroid, window_size=1
    )
    v_ca_max, w_ca_max = get_averaged_velocity_from_position(
        cond_av_max, mask_ca_max, window_size=1
    )
    v_cc_centroid, w_cc_centroid = get_averaged_velocity_from_position(
        cc_centroid, mask_cc_centroid, window_size=1
    )
    v_cc_max, w_cc_max = get_averaged_velocity_from_position(
        cross_corr_max, mask_cc_max, window_size=1
    )

    print(
        "Ca centroid: {:.4f}, {:.2f}".format(v_ca_centroid / 100, w_ca_centroid / 100)
    )
    print("Ca max: {:.4f}, {:.2f}".format(v_ca_max / 100, w_ca_max / 100))
    print(
        "Cc centroid: {:.4f}, {:.2f}".format(v_cc_centroid / 100, w_cc_centroid / 100)
    )
    print("Cc max: {:.4f}, {:.2f}".format(v_cc_max / 100, w_cc_max / 100))

    # v_tde, w_tde = im.get_tde_velocities(refx, refy, ds)

    # print("TDE: {:.4f}, {:.2f}".format(v_tde, w_tde))

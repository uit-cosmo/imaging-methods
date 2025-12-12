from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import cosmoplots as cp
import imaging_methods as im
from utils import *
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

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


def movie(
    dataset: xr.Dataset,
    interval: int = 100,
    interpolation: str = "spline16",
    file_name=None,
) -> None:
    dt = im.get_dt(dataset)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    t_dim = "time"
    refx, refy = int(dataset["refx"].item()), int(dataset["refy"].item())

    contour_cc = im.get_contour_evolution(
        dataset.cross_corr,
        method_parameters["contouring"]["threshold_factor_cc"],
    )
    contour_ca = im.get_contour_evolution(
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
    div0 = im.make_axes_locatable(ax[0])
    div1 = im.make_axes_locatable(ax[1])
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
        import os

        ani.save(file_name, writer="ffmpeg", fps=10)
        os.system(
            "gifsicle -i {} -O3 --colors 32 --lossy=150 -o {}".format(
                file_name, file_name
            )
        )

    plt.show()


def get_tde_velocities(ds, refx, refy):
    import velocity_estimation as ve

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters["2dca"]["window"] * im.get_dt(ds)
    eo.cc_options.minimum_cc_value = 0
    pd = ve.estimate_velocities_for_pixel(refx, refy, ve.CModImagingDataInterface(ds))
    return pd.vx, pd.vy


def get_simulation_data(lx, ly, theta, i):
    import os

    file_name = os.path.join(
        "synthetic_data", "data_{:.2f}_{:.2f}_{:.2f}_{}".format(lx, ly, theta, i)
    )
    if os.path.exists(file_name):
        return xr.open_dataset(file_name)
    return None


def plot_trajectories(lx, ly, theta, i):
    ds = get_simulation_data(lx, ly, theta, i)

    tdca_params = method_parameters["2dca"]
    refx, refy = tdca_params["refx"], tdca_params["refx"]
    events, average_ds = im.find_events_and_2dca(
        ds,
        tdca_params["refx"],
        tdca_params["refy"],
        threshold=tdca_params["threshold"],
        check_max=tdca_params["check_max"],
        window_size=tdca_params["window"],
        single_counting=tdca_params["single_counting"],
    )

    # movie(average_ds, file_name = "trajectories.gif")

    method = "fit"
    cond_av_max = im.compute_maximum_trajectory_da(average_ds, "cond_av", method=method)
    cross_corr_max = im.compute_maximum_trajectory_da(
        average_ds, "cross_corr", method=method
    )
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

    R, Z = (
        ds.R.isel(x=refx, y=refy).item(),
        ds.Z.isel(x=refx, y=refy).item(),
    )

    ax.scatter(R, Z)

    is_ca_high_enough = ca_max > 0.75 * np.max(ca_max)
    is_cc_high_enough = cc_max > 0.75 * np.max(cc_max)

    mask_ca_centroid = im.get_combined_mask(
        average_ds, ca_centroid, is_ca_high_enough, delta
    )
    mask_ca_max = im.get_combined_mask(
        average_ds, cond_av_max, is_ca_high_enough, delta
    )
    mask_cc_centroid = im.get_combined_mask(
        average_ds, cc_centroid, is_cc_high_enough, delta
    )
    mask_cc_max = im.get_combined_mask(
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

    ax.set_aspect("equal", adjustable="box")

    v_ca_centroid, w_ca_centroid = im.get_averaged_velocity_from_position(
        ca_centroid, mask_ca_centroid, window_size=1
    )
    v_ca_max, w_ca_max = im.get_averaged_velocity_from_position(
        cond_av_max, mask_ca_max, window_size=1
    )
    v_cc_centroid, w_cc_centroid = im.get_averaged_velocity_from_position(
        cc_centroid, mask_cc_centroid, window_size=1
    )
    v_cc_max, w_cc_max = im.get_averaged_velocity_from_position(
        cross_corr_max, mask_cc_max, window_size=1
    )
    v_tde, w_tde = get_tde_velocities(ds, refx, refy)

    print("\n ----- ESTIMATED VELOCITIES FOR PIXEL {} {} -----\n".format(refx, refy))
    print("Ca centroid: {:.4f}, {:.2f}".format(v_ca_centroid, w_ca_centroid))
    print("Ca max: {:.4f}, {:.2f}".format(v_ca_max, w_ca_max))
    print("Cc centroid: {:.4f}, {:.2f}".format(v_cc_centroid, w_cc_centroid))
    print("Cc max: {:.4f}, {:.2f}".format(v_cc_max, w_cc_max))

    print("TDE: {:.4f}, {:.2f}".format(v_tde, w_tde))


fig, ax = plt.subplots()

plot_trajectories(0.5, 2.0, 0, 0)

plt.savefig("trajectories.pdf", bbox_inches="tight")
plt.show()

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
        method_parameters.contouring.threshold_factor,
    )
    contour_ca = get_contour_evolution(
        dataset.cond_av,
        method_parameters.contouring.threshold_factor,
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


def get_synthetic_data(run_norm_radius):
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
    ds = im.run_norm_ds(ds, run_norm_radius)
    return ds


def get_positions_and_mask(
    average_ds, variable, mp: im.MethodParameters, position_method="contouring"
):
    if position_method == "contouring":
        position_da = im.get_contour_evolution(
            average_ds[variable],
            mp.contouring.threshold_factor,
            max_displacement_threshold=None,
        ).center_of_mass
    elif position_method == "max":
        position_da = im.compute_maximum_trajectory_da(
            average_ds, variable, method="parabolic"
        )
    else:
        raise NotImplementedError

    position_da, start, end = im.smooth_da(
        position_da, mp.position_filter, return_start_end=True
    )
    signal_high = (
        average_ds[variable].max(dim=["x", "y"]).values
        > 0.75 * average_ds[variable].max().item()
    )[start:end]
    mask = im.get_combined_mask(
        average_ds, position_da, signal_high, 2 * im.get_dr(average_ds)
    )

    return position_da, mask


def test_synthetic_data():
    alpha = np.pi / 8
    v_input = np.cos(alpha)
    w_input = np.sin(alpha)
    method_parameters = im.get_default_synthetic_method_params()
    method_parameters.position_filter.window_size = 1

    ds = get_synthetic_data(method_parameters.preprocessing.radius)

    tdca_params = method_parameters.two_dca
    events, average_ds = im.find_events_and_2dca(ds, tdca_params)
    # movie(average_ds)

    pos_2dca, mask_2dca = get_positions_and_mask(
        average_ds, "cond_av", method_parameters, position_method="contouring"
    )
    pos_2dcc, mask_2dcc = get_positions_and_mask(
        average_ds, "cross_corr", method_parameters, position_method="contouring"
    )
    pos_max_2dca, mask_max_2dca = get_positions_and_mask(
        average_ds, "cond_av", method_parameters, position_method="max"
    )
    pos_max_2dcc, mask_max_2dcc = get_positions_and_mask(
        average_ds, "cross_corr", method_parameters, position_method="max"
    )

    R, Z = (
        average_ds.R.isel(x=tdca_params.refx, y=tdca_params.refy).item(),
        average_ds.Z.isel(x=tdca_params.refx, y=tdca_params.refy).item(),
    )
    times = pos_2dca.time.values
    real_position = np.array([R + v_input * times, Z + w_input * times]).T

    max_error_2dca = np.max(np.abs(real_position - pos_max_2dca))
    max_error_2dcc = np.max(np.abs(real_position - pos_max_2dcc))

    print("Error 2DCA: {:.4f}".format(max_error_2dca))
    print("Error 2DCC: {:.4f}".format(max_error_2dcc))

    assert max_error_2dca < 1
    assert max_error_2dcc < 1


def test_synthetic_data_single_frame():
    alpha = np.pi / 8
    v_input = np.cos(alpha)
    w_input = np.sin(alpha)
    method_parameters = im.get_default_synthetic_method_params()
    method_parameters.position_filter.window_size = 1

    ds = get_synthetic_data(method_parameters.preprocessing.radius)

    tdca_params = method_parameters.two_dca
    events, average_ds = im.find_events_and_2dca(ds, tdca_params)
    time_index = 35

    print("time value ", average_ds.time.isel(time=time_index).item())

    variable = "cond_av"
    method = "fit"
    frame = average_ds[variable].isel(time=time_index)

    pos_r, pos_z = find_maximum_for_frame(
        frame, average_ds.R, average_ds.Z, method=method
    )

    position_da = im.compute_maximum_trajectory_da(average_ds, variable, method=method)

    fig, ax = plt.subplots()

    ax.imshow(
        frame.values,
        origin="lower",
        extent=(
            average_ds.R[0, 0],
            average_ds.R[0, -1],
            average_ds.Z[0, 0],
            average_ds.Z[-1, 0],
        ),
    )

    ax.scatter(pos_r, pos_z)

    plt.show()

from typing import Union, List

import numpy as np
from nptyping import NDArray
import xarray as xr
import superposedpulses as sp
import matplotlib.pyplot as plt
import os
import imaging_methods as im
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from blobmodel import (
    Model,
    BlobShapeImpl,
    BlobFactory,
    Blob,
    AbstractBlobShape,
)


class DeterministicBlobFactory(BlobFactory):
    def __init__(self, blobs):
        self.blobs = blobs

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain: Union[float, NDArray],
    ) -> List[Blob]:
        return self.blobs

    def is_one_dimensional(self) -> bool:
        return False


def get_blob(
    amplitude, vx, vy, posx, posy, lx, ly, t_init, theta=0, bs=BlobShapeImpl()
):
    return Blob(
        1,
        bs,
        amplitude=amplitude,
        width_prop=lx,
        width_perp=ly,
        v_x=vx,
        v_y=vy,
        pos_x=posx,
        pos_y=posy,
        t_init=t_init,
        t_drain=1e100,
        theta=theta,
        prop_shape_parameters={"lam": 0.5},
        perp_shape_parameters={"lam": 0.5},
        blob_alignment=True if theta == 0 else False,
    )


def make_0d_realization(duration, T=1e4, noise=0, dt=0.1, waiting_time=1):
    my_forcing_gen = sp.StandardForcingGenerator()
    my_forcing_gen.set_duration_distribution(lambda k: duration * np.ones(k))

    pm = sp.PointModel(waiting_time=waiting_time, total_duration=T, dt=dt)
    pm.set_pulse_shape(
        sp.ExponentialShortPulseGenerator(lam=0, tolerance=1e-20, max_cutoff=T)
    )

    pm.set_custom_forcing_generator(my_forcing_gen)

    times, signal = pm.make_realization()

    if noise != 0:
        signal = signal + np.random.normal(0, noise, len(signal))

    signal = (signal - signal.mean()) / signal.std()

    return times, signal


def make_2d_realization(
    Lx,
    Ly,
    T,
    nx,
    ny,
    dt,
    num_blobs,
    vx,
    vy,
    lx,
    ly,
    theta,
    bs,
    blob_getter="deterministic",
):
    if blob_getter == "deterministic":
        blobs = [
            get_blob(
                amplitude=np.random.exponential(),
                vx=vx,
                vy=vy,
                posx=0,
                posy=np.random.uniform(0, Ly),
                lx=lx,
                ly=ly,
                t_init=np.random.uniform(0, T),
                bs=bs,
                theta=theta,
            )
            for _ in range(num_blobs)
        ]
    else:
        blobs = [blob_getter() for _ in range(num_blobs)]

    bf = DeterministicBlobFactory(blobs)

    model = Model(
        Nx=nx,
        Ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        T=T,
        num_blobs=num_blobs,
        blob_shape=BlobShapeImpl(),
        periodic_y=False,
        t_drain=1e10,
        blob_factory=bf,
        verbose=True,
        t_init=0,
    )
    ds = model.make_realization(speed_up=True, error=1e-10)
    grid_r, grid_z = np.meshgrid(ds.x.values, ds.y.values)

    return xr.Dataset(
        {"frames": (["y", "x", "time"], ds.n.values)},
        coords={
            "R": (["y", "x"], grid_r),
            "Z": (["y", "x"], grid_z),
            "time": (["time"], ds.t.values),
        },
    )


def plot_frames(ds, t_indexes, variable="frames"):
    fig, ax = plt.subplots(
        2, 4, figsize=(4 * 2.08, 2 * 2.08), gridspec_kw={"hspace": 0.4}
    )
    for i in np.arange(8):
        axe = ax[i // 4][i % 4]
        im = axe.imshow(
            ds[variable].isel(time=int(t_indexes[i])).values,
            origin="lower",
            interpolation=None,  # "spline16",
        )
        im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
        im.set_clim(0, 2)
        axe.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
        axe.set_xlim((ds.R[0, 0], ds.R[0, -1]))
        t = ds["time"].isel(time=int(t_indexes[i])).item()
        axe.set_title(r"$t={:.1f}\,\tau_\text{{d}}$".format(t))

    return fig


def plot_frames_with_contour(average, contours, t_indexes, variable="cond_av"):
    fig, ax = plt.subplots(
        2, 4, figsize=(4 * 2.08, 2 * 2.08), gridspec_kw={"hspace": 0, "wspace": 0}
    )
    for i in np.arange(8):
        axe = ax[i // 4][i % 4]
        im = axe.imshow(
            average[variable].isel(time=int(t_indexes[i])).values,
            origin="lower",
            interpolation=None,  # "spline16",
        )
        c = contours.contours.isel(time=int(t_indexes[i])).data
        axe.plot(c[:, 0], c[:, 1], ls="--", color="black")
        im.set_extent(
            (average.R[0, 0], average.R[0, -1], average.Z[0, 0], average.Z[-1, 0])
        )
        im.set_clim(0, 2)
        axe.set_ylim((average.Z[0, 0], average.Z[-1, 0]))
        axe.set_xlim((average.R[0, 0], average.R[0, -1]))
        t = average["time"].isel(time=int(t_indexes[i])).item()
        axe.text(3, 6.5, r"$t={:.1f}\,\tau_\text{{d}}$".format(t), color="white")
        axe.set_xticks([])
        axe.set_yticks([])

    return fig


def full_analysis(
    ds,
    method_parameters: im.MethodParameters,
    suffix,
    figures_dir="integrated_tests_figures",
    do_plots=True,
    variable="cond_av",
):
    """
    Does a full analysis on imaging data, estimating a BlobParameters object containing all estimates. Plots figures
    when relevant in the provided figures_dir.
    """
    dt = im.get_dt(ds)
    dr = im.get_dr(ds)

    if do_plots:
        t_indexes = np.linspace(100, 110, num=8) / dt
        fig = plot_frames(ds, t_indexes)
        plt.savefig(
            os.path.join(figures_dir, "data_portion_{}.eps".format(suffix)),
            bbox_inches="tight",
        )

    tdca_params = method_parameters.two_dca
    refx, refy = tdca_params.refx, tdca_params.refy
    events, average_ds = im.find_events_and_2dca(
        ds,
        tdca_params.refx,
        tdca_params.refy,
        threshold=tdca_params.threshold,
        check_max=tdca_params.check_max,
        window_size=tdca_params.window,
        single_counting=tdca_params.single_counting,
    )

    contour_ds = im.get_contour_evolution(
        average_ds[variable],
        method_parameters.contouring.threshold_factor,
        max_displacement_threshold=None,
        com_method="centroid",
    )

    if do_plots:
        t_indexes = np.linspace(0, tdca_params.window, num=10)
        plot_frames_with_contour(average_ds, contour_ds, t_indexes)
        plt.savefig(
            os.path.join(figures_dir, "cond_av_{}.eps".format(suffix)),
            bbox_inches="tight",
        )

        contour_file_name = os.path.join(figures_dir, "contours_{}.gif".format(suffix))
        im.show_movie_with_contours(
            average_ds,
            contour_ds,
            apd_dataset=None,
            variable=variable,
            lims=(0, np.max(average_ds[variable].values)),
            gif_name=contour_file_name,
            interpolation="spline16",
            show=False,
        )

    signal_high = average_ds[variable].max(dim=["x", "y"]).values > 0.75
    mask = im.get_combined_mask(
        average_ds, contour_ds.center_of_mass, signal_high, 2 * dr
    )

    v_c, w_c = im.get_averaged_velocity_from_position(
        position_da=contour_ds.center_of_mass, mask=mask
    )

    fit_params = method_parameters.gauss_fit
    lx, ly, theta = im.fit_ellipse_to_event(
        average_ds.cond_av,
        tdca_params.refx,
        tdca_params.refy,
        size_penalty_factor=fit_params.size_penalty,
        aspect_ratio_penalty_factor=fit_params.aspect_penalty,
        theta_penalty_factor=fit_params.tilt_penalty,
    )
    area = contour_ds.area.sel(time=0).item()

    v_f, w_f = im.get_3tde_velocities(
        average_ds.cond_av, tdca_params.refx, tdca_params.refy
    )

    v_2tde, w_2tde = im.get_2tde_velocities(
        average_ds.cond_av, tdca_params.refx, tdca_params.refy
    )

    taud, lam = im.DurationTimeEstimator(
        im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
    ).estimate_duration_time(
        ds.frames.isel(x=tdca_params.refx, y=tdca_params.refy).values,
        im.get_dt(average_ds),
        cutoff=method_parameters.taud_estimation.cutoff,
        nperseg=method_parameters.taud_estimation.nperseg,
    )

    if do_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        im.plot_contour_at_zero(average_ds.cond_av, contour_ds, ax)

        im.plot_event_with_fit(
            average_ds,
            None,
            tdca_params.refx,
            tdca_params.refy,
            ax=ax,
            size_penalty_factor=fit_params.size_penalty,
            aspect_ratio_penalty_factor=fit_params.aspect_penalty,
            theta_penalty_factor=fit_params.tilt_penalty,
        )
        inset_ax = inset_axes(ax, width=1, height=1, loc="lower left")

        im.DurationTimeEstimator(
            im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
        ).plot_and_fit(
            ds.frames.isel(x=tdca_params.refx, y=tdca_params.refy).values,
            im.get_dt(average_ds),
            inset_ax,
            cutoff=method_parameters.taud_estimation.cutoff,
            nperseg=method_parameters.taud_estimation.nperseg,
        )
        inset_ax.get_legend().remove()
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        text = (
            r"$a={:.2f}$".format(area)
            + "\n"
            + r"$\ell_x={:.2f}\, \ell_y={:.2f}\, \theta={:.2f}$".format(lx, ly, theta)
            + "\n"
            + r"$\tau_d={:.2e}\, \lambda={:.2f}$".format(taud, lam)
            + "\n"
            + r"$Ne={}$".format(average_ds["number_events"].item())
        )
        ax.text(0.1, 0.8, text, fontsize=6, transform=ax.transAxes, color="white")

        results_file_name = os.path.join(figures_dir, "results_{}.eps".format(suffix))
        plt.savefig(results_file_name, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(3, 3, figsize=(5, 5))

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                axe = ax[i + 1][j + 1]
                values = average_ds.cond_av.isel(x=refx + i, y=refy + j).values
                axe.plot(average_ds.time, values)
                axe.vlines(
                    im.get_maximum_time(average_ds.cond_av, refx + i, refy + j),
                    0,
                    np.max(values),
                )

        tde_file_name = os.path.join(figures_dir, "tde_{}.eps".format(suffix))
        plt.savefig(tde_file_name, bbox_inches="tight")
        plt.close(fig)

    bp = im.BlobParameters(
        vx_c=v_c,
        vy_c=w_c,
        area_c=area,
        vx_2dca_tde=v_f,
        vy_2dca_tde=w_f,
        vx_tde=v_2tde,
        vy_tde=w_2tde,
        lx_f=lx,
        ly_f=ly,
        theta_f=theta,
        taud_psd=taud,
        lambda_psd=lam,
        number_events=average_ds["number_events"].item(),
    )

    return bp


def estimate_cc_tde_velocities(ds, refx, refy):
    import velocity_estimation as ve

    eo = ve.EstimationOptions()
    eo.method = ve.TDEMethod.CC

    eo.use_3point_method = True

    pd = ve.estimate_velocities_for_pixel(
        refx, refy, ve.CModImagingDataInterface(ds), eo
    )
    return pd.vx, pd.vy

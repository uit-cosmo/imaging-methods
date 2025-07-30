import phantom as ph
from test_utils import (
    make_2d_realization,
    get_blob,
    DeterministicBlobFactory,
    Model,
)
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import velocity_estimation as ve

T = 500
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.025
theta = 0
bs = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
K = T * Ly

refx, refy = 4, 4

# Method parameters
prep_radius = 1000
tdca_threshold = 2
tdca_check_max = 1
tdca_window = 150
tdca_single = True
contour_treshold = 0.3
contour_com_smooth = 3
fit_size_p = 5
fit_aspect_p = 0.2
fit_tilt_p = 0.2

taud_cutoff = 1e6
taud_nperseg = 1000
figures_dir = "integrated_tests_figures"


def test_case_a():
    ds = make_2d_realization(
        Lx, Ly, T, nx, ny, dt, K, vx=1, vy=0, lx=1, ly=1, theta=0, bs=bs
    )
    ds = ph.run_norm_ds(ds, prep_radius)
    ds.to_netcdf("test_a.nc")
    # ph.show_movie(ds.sel(time=slice(100, 110)), "frames", gif_name="test_data.gif")

    events, average_ds = ph.find_events_and_2dca(
        ds,
        refx,
        refy,
        threshold=tdca_threshold,
        check_max=tdca_check_max,
        window_size=tdca_window,
        single_counting=tdca_single,
    )

    contour_ds = ph.get_contour_evolution(
        average_ds.cond_av,
        contour_treshold,
        max_displacement_threshold=None,
    )

    contour_file_name = os.path.join(figures_dir, "contours_a.gif")
    ph.show_movie_with_contours(
        average_ds,
        contour_ds,
        "cond_av",
        lims=(0, 3),
        gif_name=contour_file_name,
        interpolation="spline16",
        show=False,
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    velocity_ds = ph.get_contour_velocity(
        contour_ds.center_of_mass,
        contour_com_smooth,
    )
    v_c, w_c = (
        velocity_ds.isel(time=slice(10, -10)).mean(dim="time", skipna=True).values
    )

    area = ph.plot_contour_at_zero(average_ds.cond_av, contour_ds, ax)

    lx, ly, theta = ph.plot_event_with_fit(
        average_ds.cond_av,
        refx,
        refy,
        ax=ax,
        size_penalty_factor=fit_size_p,
        aspect_ratio_penalty_factor=fit_aspect_p,
        theta_penalty_factor=fit_tilt_p,
    )
    inset_ax = inset_axes(ax, width=1, height=1, loc="lower left")
    v_f, w_f = ph.get_3tde_velocities(average_ds.cond_av, refx, refy)

    taud, lam, freqs = ph.DurationTimeEstimator(
        ph.SecondOrderStatistic.PSD, ph.Analytics.TwoSided
    ).plot_and_fit(
        ds.frames.isel(x=refx, y=refy).values,
        ph.get_dt(average_ds),
        inset_ax,
        cutoff=taud_cutoff,
        nperseg=taud_nperseg,
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

    results_file_name = os.path.join(figures_dir, "results_a.eps")
    plt.savefig(results_file_name, bbox_inches="tight")
    plt.close(fig)

    bp = ph.BlobParameters(
        vx_c=v_c,
        vy_c=w_c,
        area_c=area,
        vx_tde=v_f,
        vy_tde=w_f,
        lx_f=lx,
        ly_f=ly,
        theta_f=theta,
        taud_psd=taud,
        lambda_psd=lam,
        number_events=average_ds["number_events"].item(),
    )
    print(bp)

    assert np.abs(v_c - 1) < 0.1, "Wrong contour velocity"
    assert np.abs(w_c - 0) < 0.1, "Wrong contour velocity"

    assert np.abs(v_f - 1) < 0.1, "Wrong TDE velocity"
    assert np.abs(w_f - 1) < 0.1, "Wrong TDE velocity"

    assert np.abs(taud - 1) < 0.5, "Wrong duration time"

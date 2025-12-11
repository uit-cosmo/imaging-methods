import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import imaging_methods as im
from build.lib.imaging_methods import BlobParameters
from method_parameters import method_parameters
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dead_pixel_mask import get_dead_pixel_mask


def get_average(shot, refx, refy):
    file_name = os.path.join(
        "density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc"
    )
    if not os.path.exists(file_name):
        print(f"File does not exist {file_name}")
        return None
    average_ds = xr.open_dataset(file_name)
    if len(average_ds.data_vars) == 0:
        return None
    refx_ds, refy_ds = average_ds["refx"].item(), average_ds["refy"].item()
    assert refx == refx_ds and refy == refy_ds
    return average_ds


def analysis(shot, refx, refy, manager, do_plots=True):
    average_ds = get_average(shot, refx, refy)
    if average_ds is None:
        print(f"The dataset is empty for shot {shot}, ignoring...")
        return None
    gpi_ds = manager.read_shot_data(shot, data_folder="data")

    v_c, w_c, area_c = get_contour_parameters(
        shot, refx, refy, average_ds, do_plots, gpi_ds
    )
    v_2dca_tde, w_2dca_tde = get_2dca_tde_velocities(refx, refy, average_ds)
    v_tde, w_tde = get_tde_velocities(refx, refy, gpi_ds)
    lx, ly, theta = get_gaussian_fit_sizes(
        shot, refx, refy, average_ds, gpi_ds, do_plots
    )
    taud, lam = get_taud_from_psd(shot, refx, refy, gpi_ds, do_plots)
    lr, lz = plot_and_estimate_fwhm_sizes(shot, average_ds)

    return im.BlobParameters(
        vx_c=v_c,
        vy_c=w_c,
        area_c=area_c,
        vx_2dca_tde=v_2dca_tde,
        vy_2dca_tde=w_2dca_tde,
        vx_tde=v_tde,
        vy_tde=w_tde,
        lx_f=lx,
        ly_f=ly,
        lr=lr,
        lz=lz,
        theta_f=theta,
        taud_psd=taud,
        lambda_psd=lam,
        number_events=average_ds["number_events"].item(),
    )


def update_partial_analysis(
    shot, refx, refy, manager, bp: BlobParameters, do_plots=True
):
    average_ds = get_average(shot, refx, refy)
    if average_ds is None:
        print(f"The dataset is empty for shot {shot}, ignoring...")
        return None
    gpi_ds = manager.read_shot_data(shot, data_folder="data")

    v_c, w_c, area_c = get_contour_parameters(
        shot, refx, refy, average_ds, do_plots, gpi_ds
    )
    # v_2dca_tde, w_2dca_tde = get_2dca_tde_velocities(refx, refy, average_ds)
    # v_tde, w_tde = get_tde_velocities(refx, refy, gpi_ds)
    # lx, ly, theta = get_gaussian_fit_sizes(
    #    shot, refx, refy, average_ds, gpi_ds, do_plots
    # )
    # taud, lam = get_taud_from_psd(shot, refx, refy, gpi_ds, do_plots)
    # lr, lz = get_fwhm_sizes(shot, refx, refy, average_ds, do_plots)
    bp.vx_c = v_c
    bp.vy_c = w_c

    return bp


def get_tde_velocities(refx, refy, gpi_ds):
    import velocity_estimation as ve

    eo = ve.EstimationOptions()
    eo.method = ve.TDEMethod.CA

    ccf_min_lag = 1
    ca_min = 2.5
    ca_max = np.inf
    eo.cache = False

    eo.use_3point_method = True
    eo.neighbour_options = ve.NeighbourOptions(
        ccf_min_lag=ccf_min_lag, max_separation=1, min_separation=1
    )

    eo.ca_options.length_of_return = 1e-4
    eo.ca_options.distance = 0
    eo.ca_options.min_threshold = ca_min
    eo.ca_options.max_threshold = ca_max
    eo.ca_options.interpolate = True

    try:
        pd = ve.estimate_velocities_for_pixel(
            refx, refy, ve.CModImagingDataInterface(gpi_ds, get_dead_pixel_mask()), eo
        )
        vx, vy = pd.vx / 100, pd.vy / 100
    except ValueError:
        vx, vy = np.nan, np.nan
    return vx, vy


def get_taud_from_psd(shot, refx, refy, gpi_ds, do_plot):
    taud, lam = im.DurationTimeEstimator(
        im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
    ).estimate_duration_time(
        gpi_ds.frames.isel(x=refx, y=refy).values,
        im.get_dt(gpi_ds),
        cutoff=method_parameters["taud_estimation"]["cutoff"],
        nperseg=method_parameters["taud_estimation"]["nperseg"],
    )
    if do_plot:
        fig, ax = plt.subplots()
        psd_file_name = os.path.join(
            "density_scan/psds", f"psd_{shot}_{refx}{refy}.eps"
        )
        taud, lam, freqs = im.DurationTimeEstimator(
            im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
        ).plot_and_fit(
            gpi_ds.frames.isel(x=refx, y=refy).values,
            im.get_dt(gpi_ds),
            ax,
            cutoff=method_parameters["taud_estimation"]["cutoff"],
            nperseg=method_parameters["taud_estimation"]["nperseg"],
        )

        plt.savefig(psd_file_name, bbox_inches="tight")
        plt.close(fig)

    return taud, lam


def plot_and_estimate_fwhm_sizes(shot, average_ds):
    rp_fwhm, rn_fwhm, zp_fwhm, zn_fwhm = im.estimate_fwhm_sizes(average_ds)

    refx, refy = average_ds["refx"], average_ds["refy"]
    poloidal_var = average_ds.cond_av.isel(x=refx).sel(time=0).values
    r_ref = average_ds.R.isel(x=refx, y=refy).item()
    z_ref = average_ds.Z.isel(x=refx, y=refy).item()
    poloidal_pos = average_ds.Z.isel(x=refx).values - z_ref
    radial_var = average_ds.cond_av.isel(y=refy).sel(time=0).values
    radial_pos = average_ds.R.isel(y=refy).values - r_ref
    ref_val = poloidal_var[refy]

    fig, ax = plt.subplots()
    ax.plot(poloidal_pos, poloidal_var, label=r"$\Phi(Z-Z_*)$", color="blue")
    ax.plot(radial_pos, radial_var, label=r"$\Phi(R-R_*)$", color="green")

    ax.vlines([zp_fwhm, zn_fwhm], 0, ref_val, ls="--", color="blue")
    ax.vlines([rp_fwhm, rn_fwhm], 0, ref_val, ls="--", color="green")
    ax.set_xlabel(r"$R-R_*, Z-Z_*$")
    ax.legend()

    lr_lz_figname = os.path.join("density_scan/fwhm", f"fwhm_{shot}_{refx}{refy}.eps")
    plt.savefig(lr_lz_figname, bbox_inches="tight")
    plt.close(fig)

    return (rp_fwhm - rn_fwhm) / 100, (zp_fwhm - zn_fwhm) / 100


def get_gaussian_fit_sizes(shot, refx, refy, average_ds, gpi_ds, do_plots):
    fit_params = method_parameters["gauss_fit"]
    ps, pe, pt = (
        fit_params["size_penalty"],
        fit_params["aspect_penalty"],
        fit_params["tilt_penalty"],
    )
    lx, ly, theta = im.fit_ellipse_to_event(
        average_ds.cond_av,
        refx,
        refy,
        size_penalty_factor=ps,
        aspect_ratio_penalty_factor=pe,
        theta_penalty_factor=pt,
    )
    if do_plots:
        fig, ax = plt.subplots()
        fit_file_name = os.path.join(
            "density_scan/fits", f"fit_{shot}_{refx}{refy}.eps"
        )

        lx, ly, theta = im.plot_event_with_fit(
            average_ds,
            gpi_ds,
            refx,
            refy,
            ax=ax,
            fig_name=fit_file_name,
            size_penalty_factor=ps,
            aspect_ratio_penalty_factor=pe,
            theta_penalty_factor=pt,
        )
        plt.close(fig)
    return lx / 100, ly / 100, theta


def get_2dca_tde_velocities(refx, refy, average_ds):
    v_f, w_f = im.get_3tde_velocities(average_ds.cond_av, refx, refy)
    return v_f / 100, w_f / 100


def get_velocity_method_max(shot, refx, refy, average_ds, do_plots):
    return


def get_contour_parameters(shot, refx, refy, average_ds, do_plots, gpi_ds):
    contour_ds = im.get_contour_evolution(
        average_ds.cond_av,
        method_parameters["contouring"]["threshold_factor"],
        max_displacement_threshold=None,
    )
    velocity_ds = im.get_contour_velocity(
        contour_ds.center_of_mass,
        window_size=method_parameters["contouring"]["com_smoothing"],
    )
    v_c, w_c = (
        velocity_ds.isel(time=slice(10, -10)).mean(dim="time", skipna=True).values
    )
    area_c = contour_ds.area.sel(time=0).item()
    if do_plots:
        gif_file_name = os.path.join(
            "density_scan/contour_movies", f"average_contour_{shot}_{refx}{refy}.gif"
        )
        im.show_movie_with_contours(
            average_ds,
            contour_ds,
            gpi_ds,
            "cond_av",
            lims=(0, 3),
            gif_name=gif_file_name,
            interpolation="spline16",
            show=False,
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        contour_file_name = os.path.join(
            "density_scan/contours", f"contour_{shot}_{refx}{refy}.eps"
        )
        im.plot_contour_at_zero(average_ds.cond_av, contour_ds, ax, contour_file_name)
        plt.close(fig)

    return v_c / 100, w_c / 100, area_c / 100**2


def plot_results(file_suffix):
    results_file_name = os.path.join(
        "density_scan/results", f"results_{file_suffix}.json"
    )
    results = im.ResultManager.from_json(filename=results_file_name)
    if len(results.shots) == 0:
        return

    gf = np.array([r.discharge.greenwald_fraction for r in results.shots.values()])
    tauds = np.array([r.blob_params.taud_psd for r in results.shots.values()])
    v_tde = np.array([r.blob_params.vx_2dca_tde for r in results.shots.values()])
    w_tde = np.array([r.blob_params.vy_2dca_tde for r in results.shots.values()])
    lx = np.array([r.blob_params.lx_f for r in results.shots.values()])
    ly = np.array([r.blob_params.ly_f for r in results.shots.values()])

    v_c = np.array([r.blob_params.vx_c for r in results.shots.values()])
    w_c = np.array([r.blob_params.vy_c for r in results.shots.values()])
    area = np.array([r.blob_params.area_c for r in results.shots.values()])
    lx_c = np.sqrt(area / np.pi)

    fig, ax = plt.subplots(constrained_layout=True)

    ax.scatter(
        gf, v_c * tauds / lx_c, label=r"$v_\text{C} \tau_d / \ell_c$", color="black"
    )
    ax.scatter(
        gf, v_tde * tauds / lx, label=r"$v_\text{TDE} \tau_d / \ell_f$", color="green"
    )
    ax.set_xlabel(r"$f_g$")
    ax.set_ylabel(r"$v \tau_d / \ell$")

    ax.legend()
    ax.set_ylim(0, 1.1 * max((v_c * tauds / lx_c).max(), (v_tde * tauds / lx).max()))
    ax.plot()
    ratios_file_name = os.path.join("result_plots", f"ratios_{file_suffix}.eps")
    plt.savefig(ratios_file_name, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    ax[0].scatter(gf, v_c, label=r"$v_\text{C}$", color="black")
    ax[0].scatter(gf, w_c, label=r"$w_\text{C}$", color="blue")
    ax[0].scatter(gf, v_tde, label=r"$v_\text{TDE}$", color="green")
    ax[0].scatter(gf, w_tde, label=r"$w_\text{TDE}$", color="red")
    ax[0].legend()
    ax[0].set_xlabel(r"$f_g$")
    ax[0].set_ylabel(r"$v(\text{m}/\text{s})$")
    ax[0].set_ylim(-500, 1000)

    ax[1].scatter(gf, 100 * lx_c, label=r"$\ell_\text{C}$", color="black")
    ax[1].scatter(gf, 100 * lx, label=r"$\ell_x$", color="green")
    ax[1].scatter(gf, 100 * ly, label=r"$\ell_y$", color="blue")
    ax[1].legend()
    ax[1].set_xlabel(r"$f_g$")
    ax[1].set_ylabel(r"$\ell (\text{cm})$")

    vel_size_file_name = os.path.join("result_plots", f"vel_size_{file_suffix}.eps")
    plt.savefig(vel_size_file_name, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(constrained_layout=True)

    ax.scatter(gf, tauds, label=r"$\tau_\text{d}$", color="black")
    ax.set_xlabel(r"$f_g$")
    ax.set_ylabel(r"$\tau_d$")

    ax.legend()
    ax.set_ylim(0, 1.1 * np.max(tauds))
    psd_file_name = os.path.join("result_plots", f"psd_{file_suffix}.eps")
    plt.savefig(psd_file_name, bbox_inches="tight")
    plt.close(fig)


def plot_contour_figure(refx, refy):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=3.0)
    subfigure_labels = [chr(97 + i) for i in range(9)]
    ax_indx = 0

    manager = im.GPIDataAccessor(
        "/home/sosno/Git/experimental_database/plasma_discharges.json"
    )

    for shot in manager.get_shot_list():
        gpi_ds = manager.read_shot_data(shot)
        confinement_more = manager.get_discharge_by_shot(shot).comment
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        if is_hmode:
            continue
        print("Working on shot {}".format(shot))
        axe = ax[int(ax_indx / 3), ax_indx % 3]
        average_ds = get_average(shot, refx, refy)
        if len(average_ds.data_vars) == 0:
            print(f"The dataset is empty for shot {shot}, ignoring...")
            continue
        refx, refy = average_ds["refx"].item(), average_ds["refy"].item()

        contour_ds = im.get_contour_evolution(
            average_ds.cond_av,
            method_parameters["contouring"]["threshold_factor"],
            max_displacement_threshold=None,
        )
        area = im.plot_contour_at_zero(average_ds.cond_av, contour_ds, axe)

        fit_params = method_parameters["gauss_fit"]
        ps, pe, pt = (
            fit_params["size_penalty"],
            fit_params["aspect_penalty"],
            fit_params["tilt_penalty"],
        )
        lx, ly, theta = im.plot_event_with_fit(
            average_ds.cond_av,
            gpi_ds,
            refx,
            refy,
            ax=axe,
            size_penalty_factor=ps,
            aspect_ratio_penalty_factor=pe,
            theta_penalty_factor=pt,
        )

        axe.set_title(
            r"$f_{{GW}}$={:.2f}".format(
                manager.get_discharge_by_shot(shot).greenwald_fraction
            )
        )
        axe.text(
            -0.2,
            1.1,
            f"({subfigure_labels[ax_indx]})",
            transform=axe.transAxes,
            fontsize=10,
            va="top",
            ha="left",
        )

        pixel = average_ds.R[0, 1] - average_ds.R[0, 0]
        rmin, rmax, zmin, zmax = (
            average_ds.R[0, 0] - pixel / 2,
            average_ds.R[0, -1] + pixel / 2,
            average_ds.Z[0, 0] - pixel / 2,
            average_ds.Z[-1, 0] + pixel / 2,
        )
        axe.set_xlim(rmin, rmax)
        axe.set_ylim(zmin, zmax)

        inset_ax = inset_axes(axe, width=1, height=1, loc="lower left")
        #                              bbox_to_anchor=(0.1, 0.1, 0.4, 0.4),

        gpi_ds = manager.read_shot_data(shot, data_folder="../data")
        taud, lam, freqs = im.DurationTimeEstimator(
            im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
        ).plot_and_fit(
            gpi_ds.frames.isel(x=refx, y=refy).values,
            im.get_dt(average_ds),
            inset_ax,
            cutoff=method_parameters["taud_estimation"]["cutoff"],
            nperseg=method_parameters["taud_estimation"]["nperseg"],
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
        axe.text(0.1, 0.8, text, fontsize=6, transform=axe.transAxes, color="white")
        ax_indx = ax_indx + 1

    fig_name = os.path.join("result_plots", f"contours_{refx}{refy}.eps")
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close(fig)

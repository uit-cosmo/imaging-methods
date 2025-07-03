import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import phantom as ph
import cosmoplots as cp
from method_parameters import method_parameters


def get_average(shot, suffix):
    file_name = os.path.join("averages", f"average_ds_{shot}_{suffix}.nc")
    return xr.open_dataset(file_name)


def plot_movies(suffix):
    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    for shot in manager.get_shot_list():
        average_ds = get_average(shot, suffix)
        gif_name = "2d_cond_av_average_{}.gif".format(shot)
        ph.show_movie(
            average_ds,
            variable="cond_av",
            lims=(0, 3),
            gif_name=gif_name,
            show=False,
        )


def plot_velocities(suffix):
    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    fig, ax = plt.subplots()
    for shot in manager.get_shot_list():
        print("Working on shot {}".format(shot))
        ds = manager.read_shot_data(shot, None)
        average_ds = get_average(shot, suffix)
        contour_ds = ph.get_contour_evolution(
            average_ds.cond_av,
            method_parameters["contouring"]["threshold_factor"],
            max_displacement_threshold=None,
        )
        velocity_ds = ph.get_contour_velocity(
            contour_ds.center_of_mass,
            window_size=method_parameters["contouring"]["com_smoothing"],
        )
        ax.plot(
            velocity_ds.time.values, velocity_ds.values[:, 0], label="{}".format(shot)
        )

    plt.savefig("velocity_evolution.png".format(shot), bbox_inches="tight")
    fig.show()


def analysis(file_suffix=None):
    results_file_name = os.path.join("results", f"results_{file_suffix}.json")
    if os.path.exists(results_file_name):
        print(
            results_file_name
            + " exists, remove if you want to run new analysis, returning..."
        )
        return

    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    results = ph.ScanResults()
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        if is_hmode:
            continue
        print("Working on shot {}".format(shot))
        average_ds = get_average(shot, file_suffix)
        gpi_ds = manager.read_shot_data(shot, data_folder="../data")
        refx, refy = average_ds["refx"].item(), average_ds["refy"].item()

        contour_ds = ph.get_contour_evolution(
            average_ds.cond_av,
            method_parameters["contouring"]["threshold_factor"],
            max_displacement_threshold=None,
        )
        velocity_ds = ph.get_contour_velocity(
            contour_ds.center_of_mass,
            window_size=method_parameters["contouring"]["com_smoothing"],
        )
        v_c, w_c = (
            velocity_ds.isel(time=slice(10, -10)).mean(dim="time", skipna=True).values
        )
        v_c, w_c = v_c / 100, w_c / 100

        gif_file_name = os.path.join(
            "contour_movies", f"average_contour_{shot}_{file_suffix}.gif"
        )
        ph.show_movie_with_contours(
            average_ds,
            contour_ds,
            "cond_av",
            lims=(0, 3),
            gif_name=gif_file_name,
            interpolation="spline16",
            show=False,
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        contour_file_name = os.path.join(
            "contours", f"contour_{shot}_{file_suffix}.eps"
        )
        area_c = ph.plot_contour_at_zero(
            average_ds.cond_av, contour_ds, ax, contour_file_name
        )
        area_c = area_c / 100**2
        plt.close(fig)

        v_f, w_f = ph.get_3tde_velocities(average_ds.cond_av, refx, refy)
        v_f, w_f = v_f / 100, w_f / 100

        fig, ax = plt.subplots()
        fit_file_name = os.path.join("fits", f"fit_{shot}_{file_suffix}.eps")
        fit_params = method_parameters["gauss_fit"]
        ps, pe, pt = (
            fit_params["size_penalty"],
            fit_params["aspect_penalty"],
            fit_params["tilt_penalty"],
        )
        lx, ly, theta = ph.plot_event_with_fit(
            average_ds.cond_av,
            refx,
            refy,
            ax=ax,
            fig_name=fit_file_name,
            size_penalty_factor=ps,
            aspect_ratio_penalty_factor=pe,
            theta_penalty_factor=pt,
        )
        lx, ly = lx / 100, ly / 100
        plt.close(fig)

        fig, ax = plt.subplots()

        psd_file_name = os.path.join("psds", f"psd_{shot}_{file_suffix}.eps")
        taud, lam, freqs = ph.DurationTimeEstimator(
            ph.SecondOrderStatistic.PSD, ph.Analytics.TwoSided
        ).plot_and_fit(
            gpi_ds.frames.isel(x=refx, y=refy).values,
            ph.get_dt(average_ds),
            ax,
            cutoff=method_parameters["taud_estimation"]["cutoff"],
            nperseg=method_parameters["taud_estimation"]["nperseg"],
        )

        plt.savefig(psd_file_name, bbox_inches="tight")
        plt.close(fig)

        bp = ph.BlobParameters(
            vx_c=v_c,
            vy_c=w_c,
            area_c=area_c,
            vx_tde=v_f,
            vy_tde=w_f,
            lx_f=lx,
            ly_f=ly,
            theta_f=theta,
            taud_psd=taud,
            lambda_psd=lam,
            number_events=average_ds["number_events"].item(),
        )
        results.add_shot(manager.get_discharge_by_shot(shot), bp)

    results.to_json(results_file_name)


def plot_vertical_conditional_average(file_suffix):
    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    fig, ax = plt.subplots()
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        if is_hmode:
            continue
        print("Working on shot {}".format(shot))
        average_ds = get_average(shot, file_suffix)
        refx, refy = average_ds["refx"].item(), average_ds["refy"].item()
        vertical_points = average_ds.cond_av.sel(time=0).isel(x=refx).values
        poloidal_coord = (
            average_ds.Z.isel(x=refx).values - average_ds.Z.isel(x=refx, y=refy).item()
        )
        label = r"$f_{{\text{{GW}}}}={:.2f}$".format(
            manager.get_discharge_by_shot(shot).greenwald_fraction
        )
        ax.plot(poloidal_coord, vertical_points / np.max(vertical_points), label=label)

    ax.legend()
    ax.set_xlabel(r"$(Z-Z^*)/\text{cm}$")
    fig_name = os.path.join("result_plots", f"vertical_{file_suffix}.eps")
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close(fig)


def plot_results(file_suffix):
    results_file_name = os.path.join("results", f"results_{file_suffix}.json")
    results = ph.ScanResults.from_json(filename=results_file_name)

    gf = np.array([r.discharge.greenwald_fraction for r in results.shots])
    tauds = np.array([r.blob_params.taud_psd for r in results.shots])
    v_tde = np.array([r.blob_params.vx_tde for r in results.shots])
    w_tde = np.array([r.blob_params.vy_tde for r in results.shots])
    lx = np.array([r.blob_params.lx_f for r in results.shots])
    ly = np.array([r.blob_params.ly_f for r in results.shots])

    v_c = np.array([r.blob_params.vx_c for r in results.shots])
    w_c = np.array([r.blob_params.vy_c for r in results.shots])
    area = np.array([r.blob_params.area_c for r in results.shots])
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


def plot_contour_figure(suffix):
    fig, ax = plt.subplots(3, 3, figsize=(7, 7))
    plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=3.0)
    subfigure_labels = [chr(97 + i) for i in range(9)]
    ax_indx = 0

    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        if is_hmode:
            continue
        print("Working on shot {}".format(shot))
        axe = ax[int(ax_indx / 3), ax_indx % 3]
        average_ds = get_average(shot, suffix)
        refx, refy = average_ds["refx"].item(), average_ds["refy"].item()

        contour_ds = ph.get_contour_evolution(
            average_ds.cond_av,
            method_parameters["contouring"]["threshold_factor"],
            max_displacement_threshold=None,
        )
        area = ph.plot_contour_at_zero(average_ds.cond_av, contour_ds, axe)

        fit_params = method_parameters["gauss_fit"]
        ps, pe, pt = (
            fit_params["size_penalty"],
            fit_params["aspect_penalty"],
            fit_params["tilt_penalty"],
        )
        lx, ly, theta = ph.plot_event_with_fit(
            average_ds.cond_av,
            refx,
            refy,
            ax=axe,
            size_penalty_factor=ps,
            aspect_ratio_penalty_factor=pe,
            theta_penalty_factor=pt,
        )

        text = (
            r"$a={:.2f}$".format(area)
            + "\n"
            + r"$\ell_x={:.2f}\, \ell_y={:.2f}\, \theta={:.2f}$".format(lx, ly, theta)
            + "\n"
            + r"$Ne={}$".format(average_ds["number_events"].item())
        )
        axe.text(0.1, 0.8, text, fontsize=4, transform=axe.transAxes, color="white")

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

        ax_indx = ax_indx + 1

    fig_name = os.path.join("result_plots", f"contours_{suffix}.eps")
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close(fig)

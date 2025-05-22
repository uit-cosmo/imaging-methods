from phantom import (
    PlasmaDischargeManager,
    PlasmaDischarge,
    ShotAnalysis,
    find_events,
    show_movie,
    plot_event_with_fit,
    get_delays,
    get_maximum_amplitude,
    get_3tde_velocities,
    fit_psd,
    get_dt,
    ScanResults,
    get_contour_evolution,
    get_contour_velocity,
    show_movie_with_contours,
)
import numpy as np
import matplotlib.pyplot as plt
import velocity_estimation as ve
import os

manager = PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")

plot_duration_times = True
plot_movies = False
preprocess_data = False
plot_velocities = False

refx, refy = 6, 6

if preprocess_data:
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        if confinement_more == "EDA-H" or confinement_more == "ELM-free-H":
            print("Working on {}".format(shot))
            ds = manager.read_shot_data(shot, None, preprocessed=False)
            file_name = os.path.join("data", f"apd_{shot}_preprocessed.nc")
            ds.to_netcdf(file_name)

if plot_movies:
    for shot in manager.get_shot_list():
        ds = manager.read_shot_data(shot, 0.001)
        events, average, std = find_events(
            ds,
            refx,
            refy,
            threshold=2,
            check_max=1,
            window_size=60,
            single_counting=True,
        )
        gif_name = "2d_cond_av_average_{}.gif".format(shot)
        show_movie(
            std,
            variable="frames",
            #        lims=(0, np.max(average.frames.values)),
            gif_name=gif_name,
            show=False,
        )


if plot_duration_times:
    results = ScanResults()
    use_contouring = False
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        # if not is_hmode:
        # continue
        print("Working on shot {}".format(shot))
        ds = manager.read_shot_data(shot, None)
        refx, refy = 6, 5
        events, average, std = find_events(
            ds,
            refx,
            refy,
            threshold=2,
            check_max=1,
            window_size=60,
            single_counting=True,
        )
        if use_contouring:
            contour_ds = get_contour_evolution(
                average, 0.75, max_displacement_threshold=None
            )
            velocity_ds = get_contour_velocity(contour_ds.center_of_mass, sigma=3)
            v, w = (
                velocity_ds.isel(time=slice(10, -10))
                .mean(dim="time", skipna=True)
                .values
            )
            v, w = v / 100, w / 100

            area = contour_ds.area.mean(dim="time").item()
            area = area / 100**2
            lx, ly, theta = area, 0, 0

            show_movie_with_contours(
                average,
                refx,
                refy,
                contour_ds,
                "frames",
                gif_name="average_contour_{}.gif".format(shot),
                interpolation="spline16",
                show=False,
            )
        else:
            v, w = get_3tde_velocities(average)
            v, w = v / 100, w / 100

            fig, ax = plt.subplots()
            lx, ly, theta = plot_event_with_fit(
                average, ax, "average_fig_{}.png".format(shot)
            )
            lx, ly = lx / 100, ly / 100
            fig.clf()

        fig, ax = plt.subplots()

        taud, lam = fit_psd(
            ds.frames.isel(x=refx, y=refy).values,
            get_dt(ds),
            nperseg=10**3,
            ax=ax,
            cutoff_freq=1e6,
        )

        plt.savefig("psd_{}.png".format(shot), bbox_inches="tight")
        fig.clf()

        results.add_shot(
            ShotAnalysis(
                v,
                w,
                lx,
                ly,
                theta,
                taud,
                lam,
                manager.get_discharge_by_shot(shot),
            )
        )
    #    results.to_json("results_contouring.json")
    results.print_summary()

if plot_velocities:
    fig, ax = plt.subplots()
    for shot in manager.get_shot_list():
        print("Working on shot {}".format(shot))
        ds = manager.read_shot_data(shot, None)
        refx, refy = 6, 5
        events, average, std = find_events(
            ds,
            refx,
            refy,
            threshold=2,
            check_max=1,
            window_size=60,
            single_counting=True,
        )
        contour_ds = get_contour_evolution(
            average, 0.75, max_displacement_threshold=None
        )
        velocity_ds = get_contour_velocity(contour_ds.center_of_mass, sigma=1)
        ax.plot(
            velocity_ds.time.values, velocity_ds.values[:, 0], label="{}".format(shot)
        )

    plt.savefig("velocity_evolution.png".format(shot), bbox_inches="tight")
    fig.show()

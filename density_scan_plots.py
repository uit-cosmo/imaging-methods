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

refx, refy = 6, 5

if preprocess_data:
    for shot in manager.get_shot_list():
        ds = manager.read_shot_data(shot, None)
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

results = []

if plot_duration_times:
    for shot in manager.get_shot_list():
        ds = manager.read_shot_data(shot, 0.01)
        events, average, std = find_events(
            ds,
            refx,
            refy,
            threshold=2,
            check_max=1,
            window_size=60,
            single_counting=True,
        )
        v, w = get_3tde_velocities(average)

        fig, ax = plt.subplots()
        lx, ly, theta = plot_event_with_fit(
            average, ax, "average_fig_{}.png".format(shot)
        )
        fig.clf()

        fig, ax = plt.subplots()

        taud, lam = fit_psd(
            ds.frames.isel(x=refx, y=refy).values, get_dt(ds), nperseg=10**4, ax=ax
        )
        plt.savefig("psd_{}.png".format(shot), bbox_inches="tight")
        fig.clf()

        sh = ShotAnalysis(shot, v, w, lx, ly, theta, taud, lam)
        results.append(sh)

[sh.print_results() for sh in results]

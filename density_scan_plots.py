from phantom import (
    PlasmaDischargeManager,
    PlasmaDischarge,
    find_events,
    show_movie,
    plot_event_with_fit,
)
import numpy as np
import matplotlib.pyplot as plt

manager = PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")

plot_duration_times = True
plot_movies = False

refx, refy = 6, 5

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
        fig, ax = plt.subplots()
        lx, ly, theta = plot_event_with_fit(
            average, ax, "average_fig_{}.png".format(shot)
        )

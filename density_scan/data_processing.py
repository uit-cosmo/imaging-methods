import phantom as ph
import os

manager = ph.PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")

plot_duration_times = False
plot_movies = False
preprocess_data = False
plot_velocities = False
compute_and_store_conditional_averages = True

refx, refy = 6, 6

if preprocess_data:
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        if confinement_more == "EDA-H" or confinement_more == "ELM-free-H":
            print("Working on {}".format(shot))
            ds = manager.read_shot_data(shot, None, preprocessed=False)
            file_name = os.path.join("../data", f"apd_{shot}_preprocessed.nc")
            ds.to_netcdf(file_name)

if compute_and_store_conditional_averages:
    for shot in manager.get_shot_list():
        ds = manager.read_shot_data(shot, 0.001)
        events, average_ds = ph.find_events_and_2dca(
            ds,
            refx,
            refy,
            threshold=2,
            check_max=1,
            window_size=60,
            single_counting=True,
        )
        average_ds.to_netcdf("average_ds_{}.nc".format(shot))
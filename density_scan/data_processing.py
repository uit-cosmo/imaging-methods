import phantom as ph
import os
import sys
from utils import *


def preprocess_data(refx, refy):
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        print("Working on {}".format(shot))
        ds = manager.read_shot_data(
            shot, None, preprocessed=False, data_folder="../data"
        )
        file_name = os.path.join("../data", f"apd_{shot}_preprocessed.nc")
        ds.to_netcdf(file_name)


def compute_and_store_conditional_averages(refx, refy, file_suffix=None):
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        if is_hmode:
            continue
        file_name = os.path.join("averages", f"average_ds_{shot}_{file_suffix}.nc")
        if os.path.exists(file_name):
            print(file_name, " already exists, reusing...")
            continue
        ds = manager.read_shot_data(shot, None, data_folder="../data")
        events, average_ds = ph.find_events_and_2dca(
            ds,
            refx,
            refy,
            threshold=2,
            check_max=1,
            window_size=60,
            single_counting=True,
        )
        average_ds.to_netcdf(file_name)


if __name__ == "__main__":
    try:
        # Read the two integers from command-line arguments
        refx = int(sys.argv[1])
        refy = int(sys.argv[2])
        print(f"Refx: {refx}, Refy: {refy}")
    except ValueError:
        print("Error: Please ensure both arguments are valid integers.")

    suffix = f"{refx}{refy}"
    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    print("Computes 2D averages")
    compute_and_store_conditional_averages(refx, refy, file_suffix=suffix)
    print("Analyzing averages...")
    # analysis(suffix)
    print("Plotting results...")
    plot_results(suffix)
    plot_contour_figure(suffix)
    plot_fit_figure(suffix)
    plot_vertical_conditional_average(suffix)

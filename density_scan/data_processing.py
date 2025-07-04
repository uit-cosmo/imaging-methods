import phantom as ph
import os
import sys
from utils import *
from method_parameters import method_parameters


def preprocess_data():
    for shot in manager.get_shot_list():
        file_name = os.path.join("../data", f"apd_{shot}_preprocessed.nc")
        if os.path.exists(file_name):
            continue
        print("Preprocessing data for shot {}".format(shot))
        ds = manager.read_shot_data(
            shot,
            None,
            preprocessed=False,
            data_folder="../data",
            radius=method_parameters["preprocessing"]["radius"],
        )
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
        ds = manager.read_shot_data(
            shot, None, data_folder="../data", preprocessed=True
        )
        events, average_ds = ph.find_events_and_2dca(
            ds,
            refx,
            refy,
            threshold=method_parameters["2dca"]["threshold"],
            check_max=method_parameters["2dca"]["check_max"],
            window_size=method_parameters["2dca"]["window"],
            single_counting=method_parameters["2dca"]["single_counting"],
        )
        average_ds.to_netcdf(file_name)


if __name__ == "__main__":
    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    preprocess_data()

    refx, refy = method_parameters["2dca"]["refx"], method_parameters["2dca"]["refy"]
    suffix = f"{refx}{refy}"
    print(f"Refx: {refx}, Refy: {refy}")

    print("Computes 2D averages")
    compute_and_store_conditional_averages(refx, refy, file_suffix=suffix)
    print("Analyzing averages...")
    analysis(suffix)
    print("Plotting results...")
    plot_results(suffix)
    plot_contour_figure(suffix)
    plot_vertical_conditional_average(suffix)

from phantom import ResultManager
from utils import *
from method_parameters import method_parameters

import multiprocessing as mp
from functools import partial
import traceback
import logging


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


def compute_and_store_conditional_averages(shot, refx, refy, file_suffix=None):
    file_name = os.path.join("averages", f"average_ds_{shot}_{file_suffix}.nc")
    if os.path.exists(file_name):
        # print(file_name, " already exists, reusing...")
        return
    ds = manager.read_shot_data(shot, None, data_folder="../data", preprocessed=True)
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


def process_point(args, manager, results: ResultManager, force_redo=False):
    shot, refx, refy = args
    if (
        results.get_blob_params_for_shot(shot, refx, refy) is not None
        and not force_redo
    ):
        return
    try:
        # print(f"Working on shot {shot} and pixel {refx}{refy}")
        compute_and_store_conditional_averages(
            shot, refx, refy, file_suffix=f"{refx}{refy}"
        )
        bp = analysis(shot, refx, refy, manager, do_plots=False)
        if bp is None:
            return
        with mp.Lock():  # Ensure thread-safe access to results
            if shot not in results.shots:
                results.add_shot(manager.get_discharge_by_shot(shot), {})
            results.add_blob_params(shot, refx, refy, bp)
    except Exception as e:
        print(f"Issues for shot {shot}, refx={refx}, refy={refy}")
        print(traceback.format_exc())


def run_parallel():
    # Create a list of all (shot, refx, refy) combinations
    tasks = [
        (shot, refx, refy)
        for shot in manager.get_shot_list()
        for refx in range(8)
        for refy in range(10)
    ]

    # Use multiprocessing Pool to parallelize
    num_processes = mp.cpu_count()  # Use all available CPU cores
    with mp.Pool(processes=num_processes) as pool:
        pool.map(partial(process_point, manager=manager, results=results), tasks)


def run_single_thread(force_redo=False):
    for shot in manager.get_shot_list():
        for refx in range(8):
            for refy in range(10):
                try:
                    if (
                        results.get_blob_params_for_shot(shot, refx, refy) is not None
                        and not force_redo
                    ):
                        return
                    print(f"Working on shot {shot} and pixel {refx}{refy}")
                    compute_and_store_conditional_averages(
                        refx, refy, file_suffix=f"{refx}{refy}"
                    )
                    bp = analysis(shot, refx, refy, manager, do_plots=False)
                    if bp is None:
                        continue
                    if shot not in results.shots:
                        results.add_shot(manager.get_discharge_by_shot(shot), {})
                    results.add_blob_params(shot, refx, refy, bp)
                except KeyError:
                    print(f"Issues for {refx} {refy}")


if __name__ == "__main__":
    manager = ph.PlasmaDischargeManager()
    manager.load_from_json("plasma_discharges.json")
    results = ph.ResultManager.from_json("results.json")
    run_single_thread()
    results.to_json("results.json")

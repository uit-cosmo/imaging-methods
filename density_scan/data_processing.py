from imaging_methods import movie_2dca_with_contours, plot_skewness_and_flatness
from utils import *
from method_parameters import method_parameters

import multiprocessing as mp
from functools import partial
import traceback


def preprocess_data(shots):
    for shot in shots:
        file_name = os.path.join("../data", f"apd_{shot}_preprocessed.nc")
        if os.path.exists(file_name):
            continue
        print("Preprocessing data for shot {}".format(shot))
        ds = manager.read_shot_data(shot, data_folder="../data", preprocessed=False)
        ds = manager.preprocess_dataset(
            ds, radius=method_parameters["preprocessing"]["radius"]
        )
        ds.to_netcdf(file_name)


def compute_and_store_conditional_averages(shot, refx, refy):
    file_name = os.path.join(
        "density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc"
    )
    if os.path.exists(file_name):
        # print(file_name, " already exists, reusing...")
        return
    ds = manager.read_shot_data(shot, data_folder="data", preprocessed=True)
    events, average_ds = im.find_events_and_2dca(
        ds,
        refx,
        refy,
        threshold=method_parameters["2dca"]["threshold"],
        check_max=method_parameters["2dca"]["check_max"],
        window_size=method_parameters["2dca"]["window"],
        single_counting=method_parameters["2dca"]["single_counting"],
    )
    average_ds.to_netcdf(file_name)


def process_point(args, manager):
    shot, refx, refy = args
    try:
        # print(f"Working on shot {shot}, refx={refx}, refy={refy}")
        compute_and_store_conditional_averages(shot, refx, refy)
        bp = analysis(shot, refx, refy, manager, do_plots=False)
        if bp is None:
            return None
        # Return data to be added to results
        return shot, refx, refy, bp
    except Exception as e:
        print(f"Issues for shot {shot}, refx={refx}, refy={refy}")
        print(traceback.format_exc())
        return None


def run_parallel(shots, force_redo=False):
    # Create a list of all (shot, refx, refy) combinations
    tasks = []
    for shot in shots:
        for refx in [6]:  # range(9):
            for refy in [5]:  # range(10):
                if (
                    results.get_blob_params_for_shot(shot, refx, refy) is None
                    or force_redo
                ):
                    tasks.append((shot, refx, refy))

    # Use multiprocessing Pool to parallelize
    num_processes = mp.cpu_count()  # Use all available CPU cores
    num_processes = 2  # mp.cpu_count()  # Use all available CPU cores
    with mp.Pool(processes=num_processes) as pool:
        process_results = pool.map(partial(process_point, manager=manager), tasks)

    for result in process_results:
        if result is None:
            continue
        shot, refx, refy, bp = result
        if shot not in results.shots:
            results.add_shot(manager.get_discharge_by_shot(shot), {})
        results.add_blob_params(shot, refx, refy, bp)


def run_single_thread(shots, force_redo=False):
    for shot in shots:
        for refx in range(9):
            for refy in range(10):
                try:
                    if (
                        results.get_blob_params_for_shot(shot, refx, refy) is not None
                        and not force_redo
                    ):
                        return
                    print(f"Working on shot {shot} and pixel {refx}{refy}")
                    compute_and_store_conditional_averages(shot, refx, refy)
                    bp = analysis(shot, refx, refy, manager, do_plots=True)
                    if bp is None:
                        continue
                    if shot not in results.shots:
                        results.add_shot(manager.get_discharge_by_shot(shot), {})
                    results.add_blob_params(shot, refx, refy, bp)
                except KeyError:
                    print(f"Issues for {refx} {refy}")


if __name__ == "__main__":
    manager = im.PlasmaDischargeManager()
    manager.load_from_json("density_scan/plasma_discharges.json")

    results = im.ResultManager.from_json("density_scan/results.json")
    shots = [1120712027]
    preprocess_data(shots)
    #run_parallel(shots)
    #for shot in shots:
    #    movie_2dca_with_contours(shot, 6, 5)

    # run_parallel(shots, force_redo=True)
    results.to_json("density_scan/results.json")

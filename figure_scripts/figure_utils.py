import xarray as xr


def get_average(shot, refx, refy, method_parameters):
    import os
    import imaging_methods as im

    file_name = os.path.join(
        "../density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc"
    )
    if not os.path.exists(file_name):
        print(f"File does not exist {file_name}")

        manager = im.GPIDataAccessor(
            "/home/sosno/Git/experimental_database/plasma_discharges.json"
        )
        ds = manager.read_shot_data(shot, preprocessed=True, data_folder="../data")
        events, average_ds = im.find_events_and_2dca(ds, method_parameters.two_dca)
        average_ds.to_netcdf(file_name)
        return average_ds
    average_ds = xr.open_dataset(file_name)
    if len(average_ds.data_vars) == 0:
        return None
    refx_ds, refy_ds = average_ds["refx"].item(), average_ds["refy"].item()
    assert refx == refx_ds and refy == refy_ds
    return average_ds

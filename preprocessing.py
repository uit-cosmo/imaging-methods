from phantom.utils import *

shot = 1140613026

ds = xr.open_dataset("data/apd_{}.nc".format(shot))
for i in range(9):
    for j in range(10):
        if ds["frames"].values[j, i].std() < 0.01:
            ds["frames"].values[j, i] = np.nan

ds["frames"] = run_norm_ds(ds, 1000)["frames"]

t_start, t_end = get_t_start_end(ds)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.001

t_start, t_end = 0.8, 1
ds = ds.sel(time=slice(t_start, t_end))
interpolate_nans_3d(ds)
ds.to_netcdf("ds_short.nc")

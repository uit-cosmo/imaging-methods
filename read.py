from phantom.show_data import *
from phantom.utils import *

shot = 1160616025

ds = xr.open_dataset("data/phantom_{}.nc".format(shot))
# ds = xr.open_dataset("~/phantom_1160616016.nc")
# ph_1160616016 goes 1.1 to 1.6
# ph_1120921007 goes 1.35 to 1.5

# Running mean
ds["frames"] = run_norm_ds(ds, 1000)["frames"]

t_start, t_end = get_t_start_end(ds)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.001
ds = ds.sel(time=slice(t_start, t_end))

# Roll mean in space
box_size = 5
ds = ds.rolling(x=box_size, y=box_size, center=True, min_periods=1).mean()

ds = ds.coarsen(x=4).mean().coarsen(y=4).mean()
dt = get_dt(ds)

show_movie(ds, variable="frames", gif_name="output.gif")

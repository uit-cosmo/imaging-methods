import xarray as xr
from show_data import *

shot = 1120921007

ds = xr.open_dataset("~/ph_1120921007.nc")
# ds = xr.open_dataset("~/ph_1160616016.nc")
# ph_1160616016 goes 1.1 to 1.6
# ph_1120921007 goes 1.35 to 1.5
times = ds.time.values[0]
t_start = times[0]
t_end = times[len(times) - 1]
print("Data with times from {} to {}".format(t_start, t_end))

t_start = ( t_start + t_end ) / 2
t_end = t_start+0.01
ds = ds.sel(time=slice(t_start, t_end))
dt = get_dt(ds)

show_movie(dataset=ds, variable="frames", gif_name="{}_movie.gif".format(shot))
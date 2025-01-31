import xarray as xr
from show_data import *



ds = xr.open_dataset("~/ph_1120921007.nc")
# ds = xr.open_dataset("~/ph_1160616016.nc")
# ph_1160616016 goes 1.1 to 1.6
# ph_1120921007 goes 1.35 to 1.5

t_start = 1.4
t_end = 1.41
ds = ds.sel(time=slice(t_start, t_end))
dt = get_dt(ds)

show_movie(dataset=ds, variable="frames", gif_name="test.gif")
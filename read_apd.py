from phantom.show_data import *
from phantom.utils import *

shot = 1140613026

ds = xr.open_dataset("data/apd_{}.nc".format(shot))
ds = run_norm_ds(ds, 1000)

ds = ds.fillna(0)

t_start, t_end = get_t_start_end(ds)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.001
ds = ds.sel(time=slice(t_start, t_end))
ds = interpolate_nans_3d(ds)

dt = get_dt(ds)

show_movie(ds, variable="frames", gif_name="movie_apd_{}.gif".format(shot), fps=60)

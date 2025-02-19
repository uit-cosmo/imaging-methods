from xblobs import Blob
from xblobs import find_blobs
import xarray as xr
from show_data import *
from utils import *
import velocity_estimation as ve
import cosmoplots as cp
import matplotlib.pyplot as plt

shot = 1160616016

ds = xr.open_dataset("data/phantom_{}.nc".format(shot))
ds = run_norm_ds(ds, 1000)
box_size = 5
ds = ds.rolling(x=box_size, y=box_size, center=True, min_periods=1).mean()

t_start, t_end = get_t_start_end(ds)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.01
ds = ds.sel(time=slice(t_start, t_end))

blobs = find_blobs(da=ds, scale_threshold='absolute_value',
                threshold=1.3, region=40, background='flat',
                n_var='frames', t_dim='time', rad_dim='x', pol_dim='y')

blob1 = Blob(blobs, 1, n_var='frames', t_dim='time', rad_dim='x', pol_dim='y')

print(blob1)

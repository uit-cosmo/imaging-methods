from synthetic_data import *
from phantom.utils import *
from phantom.show_data import *
from phantom.cond_av import *
from phantom.utils import get_sample_data
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import numpy as np

shot = 1160616018
# ds = get_sample_data(shot, 0.2)
ds = xr.open_dataset("ds_imode_long.nc")
# ds.to_netcdf("ds_imode_long.nc")


refx, refy = 6, 5
events, average, std = find_events(
    ds, refx, refy, threshold=0.2, check_max=1, window_size=60, single_counting=True
)

# ds_corr = get_2d_corr(ds, refx, refy, delta=30*get_dt(ds))

# show_movie(ds.sel(time=slice(T/2, T/2+10)), variable="frames", lims=(0, 0.3), gif_name="data.gif")
show_movie(
    std,
    variable="frames",
    lims=(0, 1),
    gif_name="out_std.gif",
)
# show_movie(ds_corr, variable="frames", lims=(0, 1), gif_name="out.gif")

# fig, ax = plt.subplots()
# values = plot_average_blob(ds_corr, refx, refy, ax)
# plt.savefig("2d_ccf_fit.png", bbox_inches="tight")
# plt.show()

print(values)
print("LOL")

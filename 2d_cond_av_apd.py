import numpy as np
from synthetic_data import *
from phantom.show_data import show_movie
from phantom.utils import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd


ds = xr.open_dataset("ds_short.nc")

refx, refy = 6, 5

events = find_events(ds, refx, refy, threshold=2)
average = compute_average_event(events)

fig, ax = plt.subplots()

im = ax.imshow(average.sel(time=0).frames, origin="lower", interpolation="spline16")
plt.show()
print("LOL")

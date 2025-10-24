import numpy as np

import matplotlib.pyplot as plt
from imaging_methods import *
import cosmoplots as cp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

shot = 1160616027
window = 5e-5

ds = manager.read_shot_data(shot)
middle_time = ds.time.values.mean()
ds_short = ds.sel(time=slice(middle_time - window / 2, middle_time + window / 2))


show_movie(
    ds_short,
    variable="frames",
    lims=(0, ds_short.frames.max().item()),
    gif_name="apd_data.gif",
)

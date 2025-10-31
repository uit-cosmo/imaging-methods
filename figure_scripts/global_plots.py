import numpy as np
import matplotlib.pyplot as plt
from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

# Load the dataset
shot = 1160616016
manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)
ds = manager.read_shot_data(shot, preprocessed=False)

ds_new = xr.open_dataset("data/1160616016.nc")

fig, ax = plt.subplots(1, 2, figsize=(2 * 3.3, 1 * 3.3), gridspec_kw={"wspace": 0.5})
plot_skewness_and_flatness(ds, shot, fig, ax)
plt.savefig("skewness_{}.pdf".format(shot), bbox_inches="tight")
plt.show()

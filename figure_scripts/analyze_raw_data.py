import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

shot = 1150618036
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=False)

refx, refy = 6, 6

fig, ax = plt.subplots()

ax.plot(ds.time.values, ds.frames.isel(x=refx, y=refy).values)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

# Load the dataset
shot = 1110201016
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

fig, ax = plt.subplots(1, 2, figsize=(2 * 3.3, 1 * 3.3), gridspec_kw={"wspace": 0.5})
plot_skewness_and_flatness(ds, shot, fig, ax)
plt.savefig("skewness_{}.pdf".format(shot), bbox_inches="tight")
plt.show()

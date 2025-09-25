import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")

refx, refy = 6, 6


def plot_raw_for_shot(shot):
    ds = manager.read_shot_data(shot, preprocessed=False, data_folder="data/raw_shots")
    fig, ax = plt.subplots()
    ax.plot(ds.time.values, ds.frames.isel(x=refx, y=refy).values)
    ax.set_title(f"{shot}")
    plt.show()


for shot in manager.get_shot_list_by_confinement(["IWL"]):
    plot_raw_for_shot(shot)

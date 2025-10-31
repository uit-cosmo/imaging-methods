import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

refx, refy = 6, 5


def plot_raw_for_shot(shot):
    ds = manager.read_shot_data(shot, preprocessed=False, data_folder="data")
    fig, ax = plt.subplots()
    ax.plot(ds.time.values, ds.frames.isel(x=refx, y=refy).values)
    ax.set_title(f"{shot}")
    plt.show()


def plot_pdf(shot, refx, refy):
    ds = manager.read_shot_data(shot, preprocessed=False, data_folder="data")
    fig, ax = plt.subplots()
    values = ds.frames.isel(x=refx, y=refy).values
    hist, bin_edges = np.histogram(values, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ax.plot(bin_centers, hist)
    ax.set_yscale("log")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Probability")
    ax.set_title(f"{refx}{refy}")


shot = 1160616025
plot_pdf(shot, 5, 4)
plot_pdf(shot, 5, 6)
plot_pdf(shot, 5, 5)
plot_pdf(shot, 6, 5)
plot_pdf(shot, 4, 5)

plt.show()

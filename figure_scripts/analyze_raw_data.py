import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

refx, refy = 6, 5


def plot_raw_for_shot(dataset, refx, refy):
    fig, ax = plt.subplots()
    ax.plot(dataset.time.values, dataset.frames.isel(x=refx, y=refy).values)
    ax.set_title(f"{shot}")
    plt.show()


def plot_pdf(dataset, refx, refy, ax):
    values = dataset.frames.isel(x=refx, y=refy).values
    hist, bin_edges = np.histogram(values, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ax.plot(bin_centers, hist)
    ax.set_yscale("log")


def plot_multi_pdf(dataset):
    fig, ax = plt.subplots(10, 9, figsize=(35, 35))
    for x in range(9):
        for y in range(10):
            axe = ax[9-y, x]
            R, Z = dataset.R.isel(x=x, y=y).item(), dataset.Z.isel(x=x, y=y).item()
            axe.set_title("R={:.2f} Z={:.2f}".format(R, Z))
            plot_pdf(dataset, x, y, axe)
    plt.savefig("apd_raw_pdf_{}.pdf".format(shot), bbox_inches="tight")

shot = 1160927003
ds = manager.read_shot_data(shot, preprocessed=False, data_folder="data")

# plot_multi_pdf(ds)
plot_raw_for_shot(ds, 6, 5)

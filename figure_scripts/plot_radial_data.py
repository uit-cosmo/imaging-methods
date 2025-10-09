import numpy as np

import matplotlib.pyplot as plt

from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp
from scipy.signal import welch

manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")

refy = 6


def plot_raw_for_shot(shot):
    ds = manager.read_shot_data(shot, preprocessed=True, data_folder="data")
    ds = ds.sel(time=slice(0.85, 0.95))
    fig, ax = plt.subplots(figsize=(5, 5))
    for refx in range(8):
        ax.plot(
            ds.time.values,
            ds.frames.isel(x=refx, y=refy).values + 2.5 * refx,
            label=r"$R={:.2f}$ cm".format(ds.R.isel(x=refx, y=refy).item()),
        )
    ax.set_title(f"{shot}")
    ax.legend()


def plot_raw_psd_for_shot(shot):
    ds = manager.read_shot_data(shot, preprocessed=True, data_folder="data")
    ds = ds.sel(time=slice(0.85, 0.95))
    fig, ax = plt.subplots(figsize=(5, 5))
    for refx in range(8):
        base, values = signal.welch(
            ds.frames.isel(x=refx, y=refy).values, fs=1 / 5e-7, nperseg=2000
        )
        base, values = base[1:], values[1:]  # Remove zero-frequency
        base = 2 * np.pi * base  # Convert to angular frequency (rad/s)
        ax.plot(
            base,
            values,
            label=r"$R={:.2f}$ cm".format(ds.R.isel(x=refx, y=refy).item()),
        )
    ax.set_title(f"{shot}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


shot = 1150916025
plot_raw_for_shot(shot)
plot_raw_psd_for_shot(shot)
plt.show()

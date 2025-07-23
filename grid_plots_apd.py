import numpy as np

from phantom.utils import *
import matplotlib.pyplot as plt
from phantom.contours import *
from phantom import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

shot = 1160616009
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot)


def plot_multi_grid(ds):
    nx, ny = 9, 10
    fig, ax = plt.subplots(ny, nx, figsize=(4 * ny, 4 * nx))
    for x in np.arange(nx):
        for y in np.arange(ny):
            R, Z = ds.R.isel(x=x, y=y).item(), ds.Z.isel(x=x, y=y).item()
            axe = ax[ny - y - 1][x]
            data = ds.frames.isel(x=x, y=y).values
            if np.any(np.isnan(data)):
                continue

            axe.spines[["top", "bottom", "left", "right"]].set_visible(False)
            axe.set_xticks([])
            axe.set_yticks([])
            axe.set_title(r"$R={:.2f}, Z={:.2f}$".format(R, Z))

            inset_ax1 = inset_axes(axe, width=1, height=1, loc="upper left")
            inset_ax2 = inset_axes(axe, width=1, height=1, loc="upper right")
            inset_ax3 = inset_axes(axe, width=1, height=1, loc="lower left")
            inset_ax4 = inset_axes(axe, width=1, height=1, loc="lower right")

            hist, bin_edges = np.histogram(data, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            inset_ax1.plot(bin_centers, hist, color="blue")

            base, values = fppa.corr_fun(data, data, dt=5e-7)
            window = np.abs(base) < 5e-5
            inset_ax2.plot(base[window], values[window], color="blue")

            base, values = signal.welch(data, fs=1 / 5e-7, nperseg=int(2e3))
            base = 2 * np.pi * base  # Convert to angular frequency (rad/s)
            mask = np.abs(base) < 1e6
            mask[0] = False
            base, values = base[mask], values[mask]
            inset_ax3.plot(base, values, color="blue")
            inset_ax3.set_xscale("log")
            inset_ax3.set_yscale("log")

    plt.savefig("multi_plot.eps".format(shot), bbox_inches="tight")
    plt.close(fig)


def plot_pdf_grid(ds):
    nx, ny = 9, 10
    fig, ax = plt.subplots(ny, nx, figsize=(4 * ny, 4 * nx))
    for x in np.arange(nx):
        for y in np.arange(ny):
            R, Z = ds.R.isel(x=x, y=y).item(), ds.Z.isel(x=x, y=y).item()
            axe = ax[ny - y - 1][x]
            data = ds.frames.isel(x=x, y=y).values
            if np.any(np.isnan(data)):
                continue
            hist, bin_edges = np.histogram(data, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            axe.plot(bin_centers, hist, color="blue")
            axe.set_title(r"$R={:.2f}, Z={:.2f}$".format(R, Z))

    plt.savefig("pdfs_{}_raw.eps".format(shot), bbox_inches="tight")
    plt.close(fig)


def plot_acf_grid(ds):
    nx, ny = 9, 10
    fig, ax = plt.subplots(ny, nx, figsize=(4 * ny, 4 * nx))
    for x in np.arange(nx):
        for y in np.arange(ny):
            R, Z = ds.R.isel(x=x, y=y).item(), ds.Z.isel(x=x, y=y).item()
            axe = ax[ny - y - 1][x]
            data = ds.frames.isel(x=x, y=y).values
            if np.any(np.isnan(data)):
                continue
            base, values = fppa.corr_fun(data, data, dt=5e-7)
            window = np.abs(base) < 5e-5
            axe.plot(base[window], values[window], color="blue")
            axe.set_title(r"$R={:.2f}, Z={:.2f}$".format(R, Z))

    plt.savefig("acfs_{}_raw.eps".format(shot), bbox_inches="tight")
    plt.close(fig)


plot_multi_grid(ds)

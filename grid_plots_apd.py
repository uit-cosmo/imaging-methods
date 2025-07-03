import numpy as np

from phantom.utils import *
import matplotlib.pyplot as plt
from phantom.contours import *
from phantom import *

shot = 1160616009
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot)

def plot_pdf_grid(ds):
    nx, ny = 9, 10
    fig, ax = plt.subplots(ny, nx, figsize=(4*ny, 4*nx))
    for x in np.arange(nx):
        for y in np.arange(ny):
            R, Z = ds.R.isel(x=x, y=y).item(), ds.Z.isel(x=x, y=y).item()
            axe = ax[ny-y-1][x]
            data = ds.frames.isel(x=x, y=y).values
            if np.any(np.isnan(data)):
                continue
            hist, bin_edges = np.histogram(data, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
            axe.plot(bin_centers, hist, color="blue")
            axe.set_title(r"$R={:.2f}, Z={:.2f}$".format(R, Z))

    plt.savefig("pdfs_{}_raw.eps".format(shot), bbox_inches="tight")
    plt.close(fig)


def plot_acf_grid(ds):
    nx, ny = 9, 10
    fig, ax = plt.subplots(ny, nx, figsize=(4*ny, 4*nx))
    for x in np.arange(nx):
        for y in np.arange(ny):
            R, Z = ds.R.isel(x=x, y=y).item(), ds.Z.isel(x=x, y=y).item()
            axe = ax[ny-y-1][x]
            data = ds.frames.isel(x=x, y=y).values
            if np.any(np.isnan(data)):
                continue
            base, values = fppa.corr_fun(data, data, dt=5e-7)
            window = np.abs(base) < 5e-5
            axe.plot(base[window], values[window], color="blue")
            axe.set_title(r"$R={:.2f}, Z={:.2f}$".format(R, Z))

    plt.savefig("acfs_{}_raw.eps".format(shot), bbox_inches="tight")
    plt.close(fig)


plot_pdf_grid(ds)
plot_acf_grid(ds)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from imaging_methods import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp

# Load the dataset
shot = 1140613027
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

nx = ds.sizes["x"]  # Number of x pixels
ny = ds.sizes["y"]  # Number of y pixels

# Initialize array to store skewness
skewness = np.zeros((ny, nx))
kurt = np.zeros((ny, nx))

# Compute skewness for each pixel using a for loop
for x in range(nx):
    for y in range(ny):
        time_series = ds.frames.isel(x=x, y=y).values
        skewness[y, x] = skew(time_series)
        kurt[y, x] = kurtosis(time_series)

# Plot skewness using imshow
fig, ax = plt.subplots(1, 2, figsize=(2 * 3.3, 1 * 3.3), gridspec_kw={"wspace": 0.5})


def plot_image(axe, data, label):
    im = axe.imshow(data, origin="lower", cmap="viridis")
    im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
    axe.set_xlabel("R")
    axe.set_ylabel("Z")
    axe.set_title(f"{label} {shot}")
    plt.colorbar(im, ax=axe, label=label)


plot_image(ax[0], skewness, "Skewness")
plot_image(ax[1], kurt, "Kurtosis")
plt.savefig("skewness_{}.pdf".format(shot), bbox_inches="tight")
plt.show()

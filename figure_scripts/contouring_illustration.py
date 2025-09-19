import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from imaging_methods import *

# Define grid parameters
N = 5
x = np.linspace(-5, 5, N)  # x-coordinate range
y = np.linspace(-5, 5, N)  # y-coordinate range
sigma = 1.0  # Standard deviation for Gaussian

# Create 2D meshgrid
X, Y = np.meshgrid(x, y)

# Compute 2D Gaussian function
# blob = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

blob = rotated_blob((1, 2, -np.pi / 6), 0, 0, X, Y)
blob = blob[np.newaxis, :, :]

ds = xr.DataArray(
    blob,
    coords={
        "time": [0],
        "x": x,
        "y": y,
        "R": (("y", "x"), X),  # 2D auxiliary coordinate for X
        "Z": (("y", "x"), Y),  # 2D auxiliary coordinate for Y
    },
    dims=["time", "y", "x"],
    name="gaussian",
)

# Plotting
fig, ax = plt.subplots()
im = ax.imshow(
    ds.isel(time=0).values,
    origin="lower",  # Place origin at lower-left
    extent=(x.min(), x.max(), y.min(), y.max()),  # Set extent to match coordinates
    cmap="viridis",  # Color map for visualization
)
ax.set_xlabel(r"$R$")
ax.set_ylabel(r"$Z$")
ax.set_title("Tilted Gaussian")

contours = get_contour_evolution(ds, 0.3)
c = contours.contours.isel(time=0).data

ax.plot(c[:, 0], c[:, 1], color="black", ls="--")

plt.colorbar(im, ax=ax, label="Amplitude")
plt.savefig("contouring_{}.pdf".format(N), bbox_inches="tight")
plt.show()

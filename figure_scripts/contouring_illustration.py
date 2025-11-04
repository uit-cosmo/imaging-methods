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

blob = rotated_blob((1, 6, -np.pi / 6), 0, 0, X, Y)
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
for i in range(N):
    for j in range(N):
        ax.text(
            ds.R.isel(x=i, y=j).item(),
            ds.Z.isel(x=i, y=j).item(),
            "{:.2}".format(ds.isel(time=0, x=i, y=j).item()),
        )

ax.set_xlabel(r"$R$")
ax.set_ylabel(r"$Z$")
ax.set_title("Tilted Gaussian")


def indexes_to_coordinates(R, Z, indexes):
    """Convert contour indices to (r, z) coordinates using R, Z grids."""
    dx = R[0, 1] - R[0, 0]
    dy = Z[1, 0] - Z[0, 0]
    r_values = np.min(R) + indexes[:, 1] * dx
    z_values = np.min(Z) + indexes[:, 0] * dy
    return r_values, z_values


contours = get_contour_evolution(ds, 0.3)
for c in contours:
    c = indexes_to_coordinates(X, Y, c)
    ax.plot(c[0], c[1], color="black", ls="--")

# c = contours.contours.isel(time=0).data

# ax.plot(c[:, 0], c[:, 1], color="black", ls="--")

plt.colorbar(im, ax=ax, label="Amplitude")
plt.savefig("contouring_{}.pdf".format(N), bbox_inches="tight")
plt.show()

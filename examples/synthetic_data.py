import numpy as np
import imaging_methods as im
import xarray as xr
from blobmodel import (
    Model,
    show_model,
    DefaultBlobFactory,
)
import matplotlib.pyplot as plt
from matplotlib import animation

method_parameters = im.get_default_synthetic_method_params()
method_parameters.two_dca.refx = 8
method_parameters.two_dca.refy = 8


# First we do a realization of the two-dimensional process. This is done with the blobmodel package. For more info on
# this package, consult its documentation at https://blobmodel.readthedocs.io/en/latest/?badge=latest
bf = DefaultBlobFactory(
    vx_parameter=1,  # x-component of blob velocity
    vy_parameter=0,  # y-component of blob velocity
)
dt = 0.1  # Sampling time

model = Model(
    Nx=16,  # number of pixels in the x-direction
    Ny=16,  # number of pixels in the y-direction
    Lx=8,  # Domain length in the x-direction
    Ly=8,  # Domain length in the y-direction
    dt=dt,  # Sampling time
    T=1000,  # Total signal duration
    num_blobs=1000,  # Number of blobs
    periodic_y=True,  # Periodicity
    t_drain=1e10,  # No blob decay
    blob_factory=bf,
)

ds = model.make_realization(speed_up=True, error=1e-10)

# Optionally, visualize synthetic data
show_model(ds.sel(t=slice(100, 110)), gif_name="example.gif")

grid_r, grid_z = np.meshgrid(ds.x.values, ds.y.values)

# Set the data in the same format as APD data
ds = xr.Dataset(
    {"frames": (["y", "x", "time"], ds.n.values)},
    coords={
        "R": (["y", "x"], grid_r),
        "Z": (["y", "x"], grid_z),
        "time": (["time"], ds.t.values),
    },
)

# Normalize the data with a running normalization so that it has vanishing mean and unity standard deviation
ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)

# 2DCA algorithm
events, average_ds = im.find_events_and_2dca(ds, method_parameters.two_dca)

# Contouring
contour_ds = im.get_contour_evolution(
    average_ds.cond_av,
    threshold_factor=method_parameters.contouring.threshold_factor,
)

fig, ax = plt.subplots(figsize=(5, 5))

R, Z = average_ds.R.values, average_ds.Z.values
maxx = average_ds.cond_av.max().item()

def animate_2d(i):
    arr = average_ds.cond_av.isel(time=i).values
    im.set_data(arr / maxx)
    im.set_clim(0, 1)
    c = contour_ds.contours.isel(time=i).data
    line[0].set_data(c[:, 0], c[:, 1])

    time = average_ds.time[i]
    tx.set_text(r"$t/\tau_\text{{d}}=\,{:.1f}$".format(time))


time = average_ds.time[0]
tx = ax.set_title(r"$t/\tau_\text{{d}}=\,{:.1f}$".format(time))
extent = (R[0, 0], R[0, -1], Z[0, 0], Z[-1, 0])
im = ax.imshow(average_ds.cond_av.isel(time=0).values / maxx, origin="lower", interpolation=None, extent=extent)
line = ax.plot([], [], ls="--", color="black")

ani = animation.FuncAnimation(
    fig, animate_2d, frames=average_ds.time.values.size, interval=100
)

ani.save("cond_av.gif", writer="ffmpeg", fps=60)

plt.show()

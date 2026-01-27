import numpy as np
import imaging_methods as im
import xarray as xr
from blobmodel import (
    Model,
    show_model,
    DefaultBlobFactory,
)

method_parameters = im.get_default_synthetic_method_params()
method_parameters.two_dca.refx = 4
method_parameters.two_dca.refy = 4

bf = DefaultBlobFactory(
    vx_parameter=1,  # x-component of blob velocity
    vy_parameter=0,  # y-component of blob velocity
)

model = Model(
    Nx=8,  # number of pixels in the x-direction
    Ny=8,  # number of pixels in the y-direction
    Lx=8,  # Domain length in the x-direction
    Ly=8,  # Domain length in the y-direction
    dt=0.1,  # Sampling time
    T=1000,  # Total signal duration
    num_blobs=1000,  # Number of blobs
    periodic_y=True,  # Periodicity
    t_drain=1e10,  # No blob decay
    blob_factory=bf,
)
ds = model.make_realization(speed_up=True, error=1e-10)

# visualize synthetic data
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
ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)

# 2DCA algorithm
events, average_ds = im.find_events_and_2dca(ds, method_parameters.two_dca)

# Contouring
contour_ds = im.get_contour_evolution(
    average_ds.cond_av,
    threshold_factor=method_parameters.contouring.threshold_factor,
)

# visualize conditional averaging with contours
im.show_movie_with_contours(
    average_ds,
    contour_ds,
    apd_dataset=None,
    variable="cond_av",
    lims=(0, 3),
    gif_name="cond_av.gif",
    interpolation=None,
    show=False,
)

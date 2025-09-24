from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import imaging_methods as im
import xarray as xr


from blobmodel import (
    Model,
    BlobShapeImpl,
    show_model,
    DefaultBlobFactory,
    BlobFactory,
    Blob,
    AbstractBlobShape,
)


T = 1000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
num_blobs = 10_0

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
    num_blobs=10_000,  # Number of blobs
    periodic_y=False,  # No periodicity
    t_drain=1e10,  # No blob decay
    blob_factory=bf,
)
ds = model.make_realization(speed_up=True, error=1e-10)

# visualize synthetic data
show_model(ds.sel(t=slice(100, 110)), gif_name="example.gif")

grid_r, grid_z = np.meshgrid(ds.x.values, ds.y.values)
ds = xr.Dataset(
    {"frames": (["y", "x", "time"], ds.n.values)},
    coords={
        "R": (["y", "x"], grid_r),
        "Z": (["y", "x"], grid_z),
        "time": (["time"], ds.t.values),
    },
)

events, average_ds = im.find_events_and_2dca(
    ds,
    refx=4,  # reference pixel
    refy=4,  # reference pixel
    threshold=0.2,  # Threshold
    check_max=1,  # event should be larger than nearest neighbours
    window_size=50,  # window size
    single_counting=True,
)

contour_ds = im.get_contour_evolution(
    average_ds.cond_av,
    threshold_factor=0.5,
    max_displacement_threshold=None,
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

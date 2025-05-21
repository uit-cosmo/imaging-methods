from phantom.show_data import *
from phantom.cond_av import *
from phantom.contours import *
from realizations import make_2d_realization
from blobmodel import BlobShapeEnum, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np

num_blobs = 500
T = 2000
Lx = 10
Ly = 10
aspect_ratio = 1 / 3 # lx/ly
lx = np.sqrt(aspect_ratio)
ly = 1/np.sqrt(aspect_ratio)
nx = 8
ny = 8
dt = 0.1
vx = 1
vy = 0
theta = -np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

ds = make_2d_realization(Lx, Ly, T, nx, ny, dt, num_blobs, vx, vy, lx, ly, theta, bs)

# show_movie(ds.sel(time=slice(10, 20)), lims=(0, 0.3))

refx, refy = 4, 4
events, average, std = find_events(
    ds, refx, refy, threshold=0.2, check_max=1, window_size=30, single_counting=True
)

contours_da = get_contour_evolution(average, threshold_factor=0.5)
show_movie_with_contours(average, refx, refy, contours_da, "frames", gif_name="synthetic_2dca_av_8.gif", lims=(0, 0.3))


if use_contouring:
    contour_ds = get_contour_evolution(
        average, 0.75, max_displacement_threshold=None
    )
    velocity_ds = get_contour_velocity(contour_ds.center_of_mass, sigma=3)
    v, w = (
        velocity_ds.isel(time=slice(10, -10))
        .mean(dim="time", skipna=True)
        .values
    )
    v, w = v / 100, w / 100

    area = contour_ds.area.mean(dim="time").item()
    area = area / 100 ** 2
    lx, ly, theta = area, 0, 0

    show_movie_with_contours(
        average,
        refx,
        refy,
        contour_ds,
        "frames",
        gif_name="average_contour_{}.gif".format(shot),
        interpolation="spline16",
        show=False,
    )
else:
    v, w = get_3tde_velocities(average)
    v, w = v / 100, w / 100

    fig, ax = plt.subplots()
    lx, ly, theta = plot_event_with_fit(
        average, ax, "average_fig_{}.png".format(shot)
    )
    lx, ly = lx / 100, ly / 100
    fig.clf()

fig, ax = plt.subplots()

taud, lam = fit_psd(
    ds.frames.isel(x=refx, y=refy).values,
    get_dt(ds),
    nperseg=10 ** 3,
    ax=ax,
    cutoff_freq=1e6,
)



from phantom.show_data import *
from phantom.cond_av import *
from phantom.contours import *
from realizations import make_2d_realization
from blobmodel import BlobShapeEnum, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np
import cosmoplots as cp

num_blobs = 500
T = 2000
Lx = 10
Ly = 10
aspect_ratio = 1  # lx/ly
lx = np.sqrt(aspect_ratio)
ly = 1 / np.sqrt(aspect_ratio)
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

contours_ds = get_contour_evolution(average, threshold_factor=0.5)
# show_movie_with_contours(average, refx, refy, contours_ds, "frames", lims=(0, 0.3))


velocity_ds = get_contour_velocity(contours_ds.center_of_mass, window_size=11)
fig, ax = cp.figure_multiple_rows_columns(1, 1)
ax = ax[0]

ax.plot(velocity_ds.time, velocity_ds.values[:, 0], label=r"$v_c$")
tmin, tmax = velocity_ds.time.max().item(), velocity_ds.time.min().item()
ax.hlines(vx, tmin, tmax, label=r"$v$", ls="--")
ax.plot(velocity_ds.time, velocity_ds.values[:, 1], label=r"$w_c$")
ax.hlines(vy, tmin, tmax, label=r"$w$", ls="--")
ax.legend()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$v, w$")

plt.show()

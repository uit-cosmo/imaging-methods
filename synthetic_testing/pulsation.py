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
aspect_ratio = 1 # lx/ly
lx = np.sqrt(aspect_ratio)
ly = 1/np.sqrt(aspect_ratio)
nx = 8
ny = 8
dt = 0.1
vx = 1
vy = 0
theta = -np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
refx, refy = 4, 4


number_of_pixels = np.arange(4, 16, step=2)
fig, ax = cp.figure_multiple_rows_columns(1, 1)
ax = ax[0]

for n in number_of_pixels:
    refx, refy = int(n/2), int(n/2)
    pixel_size = Lx / n
    ds = make_2d_realization(Lx, Ly, T, n, n, dt, num_blobs, vx, vy, lx, ly, theta, bs)
    events, average, std = find_events(
        ds, refx, refy, threshold=0.2, check_max=1, window_size=30, single_counting=True
    )
    contours_ds = get_contour_evolution(average, threshold_factor=0.5)
#    show_movie_with_contours(average, refx, refy, contours_ds, "frames", lims=(0, 0.3))
    velocity_ds = get_contour_velocity(contours_ds.center_of_mass, window_size=5, window_type="gaussian")
#    velocity_ds = get_contour_velocity_gaussian(contours_ds.center_of_mass, sigma=3)
    ax.plot(velocity_ds.time, velocity_ds.values[:, 0], label=r"$\Delta x/\ell={:.2f}$".format(pixel_size/lx))

ax.legend()
plt.show()



from synthetic_data import *
from phantom.utils import *
from phantom.show_data import *
from phantom.cond_av import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import numpy as np


import numpy as np

# Import package to compute level set
from skimage import measure

# import function to calculate area of closed curve
from shapely.geometry import Polygon, Point

# import function to compute convex hull of polygon
from scipy.spatial import ConvexHull


def get_blob(vx, vy, posx, posy, lx, ly, t_init, theta, bs=BlobShapeImpl()):
    return Blob(
        1,
        bs,
        amplitude=1,
        width_prop=lx,
        width_perp=ly,
        v_x=vx,
        v_y=vy,
        pos_x=posx,
        pos_y=posy,
        t_init=t_init,
        t_drain=1e100,
        theta=theta,
        blob_alignment=True if theta == 0 else False,
    )


num_blobs = 5
T = 20
Lx = 5
Ly = 5
lx = 1
ly = 1
nx = 64
ny = 64
vx = 1
vy = 0
theta = 0
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

blobs = [
    get_blob(
        vx=vx,
        vy=vy,
        posx=np.random.uniform(0, Lx),
        posy=np.random.uniform(0, Ly),
        lx=lx,
        ly=ly,
        t_init=np.random.uniform(0, T),
        bs=bs,
        theta=theta,
    )
    for _ in range(num_blobs)
]

rp = RunParameters(T=T, lx=Lx, ly=Ly, nx=nx, ny=ny)
bf = DeterministicBlobFactory(
    [get_blob(vx, vy, 0, 2.5, lx, ly, t_init=10, bs=bs, theta=theta)]
)

ds = make_2d_realization(rp, bf)

refx, refy = 32, 32
events, average, std = find_events(
    ds, refx, refy, threshold=0.2, check_max=1, window_size=60, single_counting=True
)
e = events[0]

contours_list = []
for t in e.time.values:
    frame = e.sel(time=t).frames.values
    contours = measure.find_contours(frame, np.max(frame) / 2)
    contour = (
        contours[0] if len(contours) > 0 else np.array([])
    )  # Default to first contour
    if len(contours) > 1:
        contour = max(contours, key=len)  # Select longest contour

    contours_list.append(contour)


def indexes_to_coordinates(R, Z, indexes):
    dx = R[0, 1] - R[0, 0]
    dy = Z[1, 0] - Z[0, 0]
    r_values = np.min(R) + indexes[:, 0] * dx
    z_values = np.min(Z) + indexes[:, 1] * dy
    return r_values, z_values


max_points = max(len(c) for c in contours_list) if contours_list else 1
contour_data = np.full(
    (len(e.time.values), max_points, 2), np.nan
)  # Initialize with NaN
for t, contour in enumerate(contours_list):
    n_points = len(contour)
    r_values, z_values = indexes_to_coordinates(e.R.values, e.Z.values, contour)
    if n_points > 0:
        contour_data[t, :n_points, :] = np.stack(
            (r_values, z_values), axis=-1
        )  # Store (x, y) points

contours_da = xr.DataArray(
    contour_data,
    dims=("time", "point_idx", "coord"),
    coords={"time": e.time.values, "coord": ["x", "y"]},
)

show_movie_with_contours(
    events[0], refx, refy, contours_da, "frames", interpolation="spline16"
)

fig, ax = plt.subplots()


t = 0
extent = (e.R[0, 0], e.R[0, -1], e.Z[0, 0], e.Z[-1, 0])
ax.imshow(e.sel(time=t).frames.values, extent=extent)
ax.scatter(
    e.R.isel(x=refx, y=refy).item(), e.Z.isel(x=refx, y=refy).item(), color="black"
)

peak = e.frames.sel(time=t).isel(x=refx, y=refy).item()
ref_pixel = Point(e.R.isel(x=refx, y=refy).item(), e.Z.isel(x=refx, y=refy).item())

contour = measure.find_contours(e.frames.sel(time=t).values, peak / 2)
r_values, z_values = indexes_to_coordinates(e.R.values, e.Z.values, contour[0])
ax.plot(r_values, z_values, ls="--", color="black")

polygon = Polygon(np.array([r_values, z_values]).T)  # Polygon object
convex = ConvexHull(np.array([r_values, z_values]).T)

length = polygon.length  # float
convexity_deficiency = abs((convex.volume - polygon.area) / convex.volume)  # float

print(convex.area, convex.volume)

print("LOL")

plt.show()

from synthetic_data import *
from phantom.utils import *
from phantom.show_data import *
from phantom.cond_av import *
from phantom.contours import *
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


def get_blob(amplitude, vx, vy, posx, posy, lx, ly, t_init, theta, bs=BlobShapeImpl()):
    return Blob(
        1,
        bs,
        amplitude=amplitude,
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


num_blobs = 50
T = 200
Lx = 5
Ly = 5
lx = 0.2
ly = 0.6
nx = 32
ny = 32
vx = 0
vy = -1
theta = np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

blobs = [
    get_blob(
        amplitude=np.random.exponential(),
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
bf = DeterministicBlobFactory(blobs)

ds = make_2d_realization(rp, bf)

refx, refy = 16, 16
events, average, std = find_events(
    ds, refx, refy, threshold=0.2, check_max=1, window_size=30, single_counting=True
)
e = events[0]

contours_da = get_contour_evolution(events[0])

for e in events:
    contours_ds = get_contour_evolution(e, max_displacement_threshold=0.2)
    if contours_ds is None:
        continue
    velocity = get_contour_velocity(contours_ds.center_of_mass)
    avg_velocity = velocity.mean(dim="time", skipna=True)
    print(avg_velocity)
    show_movie_with_contours(
        e,
        refx,
        refy,
        contours_ds,
        "frames",
        interpolation=None,
        gif_name="e{}.gif".format(e["event_id"].item(), show=False),
    )

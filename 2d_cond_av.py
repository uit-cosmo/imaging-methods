import numpy as np
from synthetic_data import *
from phantom.show_data import show_movie
from phantom.utils import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

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


num_blobs = 1000
T = 2000
Lx = 3
Ly = 3
lx = 0.5
ly = 1.5
nx = 20
ny = 20
vx = 1
vy = 1
theta = -np.pi/4
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

blobs = [get_blob(vx=vx, vy=vy, posx=np.random.uniform(0, Lx), posy=Ly, lx=lx, ly=ly, t_init=np.random.uniform(0, T), bs=bs, theta=theta) for _ in range(num_blobs)]

rp = RunParameters(T=T, lx=Lx, ly=Ly, nx=nx, ny=ny)
bf = DeterministicBlobFactory(blobs)

ds = make_2d_realization(rp, bf)

refx, refy = 10, 10

events = find_events(ds, refx, refy, threshold=0.2)
average = compute_average_event(events)

fig, ax = plt.subplots()

im = ax.imshow(average.sel(time=0).frames, origin="lower", interpolation="spline16")
plt.show()
print("LOL")
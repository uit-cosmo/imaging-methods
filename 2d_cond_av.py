from synthetic_data import *
from phantom.utils import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import numpy as np


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
nx = 5
ny = 5
vx = 1
vy = -1
theta = -np.pi / 4
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
bf = DeterministicBlobFactory(blobs)

# ds = make_2d_realization(rp, bf)
shot = 1160616025

def get_sample_data(shot, window):
    ds = xr.open_dataset("data/apd_{}.nc".format(shot))
    ds["frames"] = run_norm_ds(ds, 1000)["frames"]

    t_start, t_end = get_t_start_end(ds)
    print("Data with times from {} to {}".format(t_start, t_end))

    t_start = (t_start + t_end) / 2
    t_end = t_start + window
    ds = ds.sel(time=slice(t_start, t_end))
    interpolate_nans_3d(ds)
    return ds

ds = xr.open_dataset("data/ds_short.nc")
#ds = get_sample_data(shot, 0.1)
#ds.to_netcdf("data_tmp.nc")

for refx in [5, 6, 7]:
    for refy in [3, 4, 5, 6, 7]:
        events = find_events(ds, refx, refy, threshold=0.2, check_max=2)
        average = compute_average_event(events)

        fig, ax = plt.subplots()
        elispe = plot_average_blob(average, refx, refy, ax)
        plt.savefig("CA{}{}".format(refx, refy), bbox_inches="tight")



plt.show()
print("LOL")

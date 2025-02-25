import xblobs as xb
import xarray as xr
from show_data import *
from utils import *
import velocity_estimation as ve
import cosmoplots as cp
import matplotlib.pyplot as plt
from synthetic_data import *


def get_blob(vx, vy, posx, posy, t_init):
    return Blob(
        1,
        BlobShapeImpl(),
        amplitude=1,
        width_prop=1,
        width_perp=1,
        v_x=vx,
        v_y=vy,
        pos_x=posx,
        pos_y=posy,
        t_init=t_init,
        t_drain=1e100,
    )


def get_random_blob():
    vx = np.random.uniform(0, 1)
    vy = np.random.uniform(-1, 1)
    posx = 0
    posy = np.random.uniform(0, 10)
    t_init = np.random.uniform(0, 10)
    return get_blob(vx, vy, posx, posy, t_init)


N = 10
blobs = [get_random_blob() for _ in range(N)]
ds = make_2d_realization(RunParameters(T=20), DeterministicBlobFactory(blobs))

blobs = xb.find_blobs(
    da=ds,
    scale_threshold="absolute_value",
    threshold=0.3,
    region=0,
    background="flat",
    n_var="frames",
    t_dim="time",
    rad_dim="x",
    pol_dim="y",
)

print("Detected {} blobs".format(blobs.blob_labels.number_of_blobs))

show_labels(blobs, fps=50)

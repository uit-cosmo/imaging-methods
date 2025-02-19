import xblobs as xb
import xarray as xr
from show_data import *
from utils import *
import velocity_estimation as ve
import cosmoplots as cp
import matplotlib.pyplot as plt
from synthetic_data import *


def get_blob(vx, vy, posx, posy):
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
        t_init=0,
        t_drain=1e100,
    )


blobs = [get_blob(1, 1, 0, 0), get_blob(1, 1, 0, 5)]

ds = make_2d_realization(RunParameters(), DeterministicBlobFactory(blobs))

blobs = xb.find_blobs(
    da=ds,
    scale_threshold="absolute_value",
    threshold=0.2,
    region=0,
    background="flat",
    n_var="n",
    t_dim="t",
    rad_dim="x",
    pol_dim="y",
)

print("Detected {} blobs".format(blobs.blob_labels.number_of_blobs))

show_labels(blobs)

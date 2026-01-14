import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp
import numpy as np
import xarray as xr

from imaging_methods import movie_dataset
from scan_utils import *

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

method_parameters = im.get_default_synthetic_method_params()

data_file = "barberpole_data.npz"
data = "data_cc"

T = 5000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 5000
NSR = 0.1


def get_simulation_data(i):
    file_name = os.path.join("synthetic_data", "{}_{}".format(data, i))
    if os.path.exists(file_name):
        return xr.open_dataset(file_name)
    alpha = 0  #  np.random.uniform(-np.pi / 4, np.pi / 4)
    vx_input, vy_input = np.cos(alpha), np.sin(alpha)
    ds = im.make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_input,
        lx=1,
        ly=1,
        theta=0,
        bs=bs,
    )
    ds_mean = ds.frames.mean().item()
    ds = ds.assign(
        frames=ds["frames"] + ds_mean * NSR * np.random.random(ds.frames.shape)
    )

    amplitude = 1.0  # Wave amplitude
    wavelength = 20.0  # Wavelength in x units (e.g., meters)
    period = 2.0  # Period in time units (e.g., seconds)
    wave_speed = wavelength / period  # Speed = lambda / T

    # Step 3: Create the propagating wave DataArray
    # Simple 1D wave propagating in +x direction: A * sin(2π (x/λ - t/T))
    # Use xarray broadcasting: expand to match (time, x, y)
    k = 2 * np.pi / wavelength  # Wave number
    omega = 2 * np.pi / period  # Angular frequency
    wave = amplitude * np.sin(k * ds.y - omega * ds.time)  # Shape: (time, x)
    wave = wave.expand_dims(x=ds.x)  # Broadcast to (time, x, y)

    # Step 4: Add the wave to the original dataset (creates a new variable)
    ds["frames"] = ds["frames"] + wave

    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    ds["v_input"] = vx_input
    ds["w_input"] = vy_input
    ds.to_netcdf(file_name)
    return ds


ds = get_simulation_data(0)
movie_dataset(ds)

from typing import Union, List

import numpy as np
from nptyping import NDArray
import xarray as xr
import superposedpulses as sp

from blobmodel import (
    Model,
    BlobShapeImpl,
    BlobFactory,
    Blob,
    AbstractBlobShape,
)


class DeterministicBlobFactory(BlobFactory):
    def __init__(self, blobs):
        self.blobs = blobs

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain: Union[float, NDArray],
    ) -> List[Blob]:
        return self.blobs

    def is_one_dimensional(self) -> bool:
        return False


def get_blob(
    amplitude, vx, vy, posx, posy, lx, ly, t_init, theta=0, bs=BlobShapeImpl()
):
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


def make_0d_realization(duration, T=1e4, noise=0, dt=0.1, waiting_time=1):
    my_forcing_gen = sp.StandardForcingGenerator()
    my_forcing_gen.set_duration_distribution(lambda k: duration * np.ones(k))

    pm = sp.PointModel(waiting_time=waiting_time, total_duration=T, dt=dt)
    pm.set_pulse_shape(
        sp.ExponentialShortPulseGenerator(lam=0, tolerance=1e-20, max_cutoff=T)
    )

    pm.set_custom_forcing_generator(my_forcing_gen)

    times, signal = pm.make_realization()

    if noise != 0:
        signal = signal + np.random.normal(0, noise, len(signal))

    signal = (signal - signal.mean()) / signal.std()

    return times, signal


def make_2d_realization(Lx, Ly, T, nx, ny, dt, num_blobs, vx, vy, lx, ly, theta, bs):
    blobs = [
        get_blob(
            amplitude=1,
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

    bf = DeterministicBlobFactory(blobs)

    model = Model(
        Nx=nx,
        Ny=ny,
        Lx=lx,
        Ly=ly,
        dt=dt,
        T=T,
        num_blobs=num_blobs,
        blob_shape=BlobShapeImpl(),
        periodic_y=False,
        t_drain=1e10,
        blob_factory=bf,
        verbose=False,
        t_init=0,
    )
    ds = model.make_realization(speed_up=True, error=1e-10)
    grid_r, grid_z = np.meshgrid(ds.x.values, ds.y.values)

    return xr.Dataset(
        {"frames": (["y", "x", "time"], ds.n.values)},
        coords={
            "R": (["y", "x"], grid_r),
            "Z": (["y", "x"], grid_z),
            "time": (["time"], ds.t.values),
        },
    )

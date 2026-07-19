from typing import Union, List

import numpy as np
import xarray as xr
from nptyping import NDArray

from blobmodel import (
    Model,
    BlobShapeImpl,
    BlobFactory,
    Blob,
    AbstractBlobShape,
)

__all__ = [
    "DeterministicBlobFactory",
    "get_blob",
    "make_2d_realization",
]


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
    amplitude,
    vx,
    vy,
    posx,
    posy,
    lx,
    ly,
    t_init,
    theta=0,
    bs=BlobShapeImpl(),
    taup=1e100,
):
    return Blob(
        1,
        bs,
        amplitude=amplitude,
        width_p=lx,
        width_s=ly,
        v_x=vx,
        v_y=vy,
        pos_x0=posx,
        pos_y0=posy,
        t_init=t_init,
        t_drain=taup,
        theta=theta,
        shape_parameters_p={"lam": 0.5},
        shape_parameters_s={"lam": 0.5},
        blob_alignment=True if theta == 0 else False,
    )


def make_2d_realization(
    Lx,
    Ly,
    T,
    nx,
    ny,
    dt,
    num_blobs,
    vx=1,
    vy=0,
    lx=1,
    ly=1,
    theta=0,
    bs=BlobShapeImpl(),
    blob_getter="deterministic",
    periodic_y=True,
    t_drain=1e10,
):
    if blob_getter == "deterministic":
        blobs = [
            get_blob(
                amplitude=np.random.exponential(),
                vx=vx,
                vy=vy,
                posx=0,
                posy=np.random.uniform(0, Ly),
                lx=lx,
                ly=ly,
                t_init=np.random.uniform(0, T),
                bs=bs,
                theta=theta,
            )
            for _ in range(num_blobs)
        ]
    else:
        blobs = [blob_getter() for _ in range(num_blobs)]

    bf = DeterministicBlobFactory(blobs)

    model = Model(
        Nx=nx,
        Ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        T=T,
        num_blobs=num_blobs,
        blob_shape=bs,
        periodic_y=periodic_y,  # Set to true so that high vertical velocities makes sense
        t_drain=t_drain,
        blob_factory=bf,
        verbose=True,
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

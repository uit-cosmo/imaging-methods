from typing import Union, List

from nptyping import NDArray

from synthetic_data import *
import json
import numpy as np

from blobmodel import (
    Model,
    DefaultBlobFactory,
    BlobShapeImpl,
    BlobFactory,
    Blob,
    AbstractBlobShape,
    DistributionEnum,
)
import matplotlib as mpl
import cosmoplots
import velocity_estimation as ve


class RunParameters:
    def __init__(
        self,
        nx=32,
        ny=32,
        lx=10,
        ly=10,
        dt=0.1,
        T=10,
        num_blobs=1,
        periodic_y=False,
        t_drain=1e100,
    ):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dt = dt
        self.T = T
        self.num_blobs = num_blobs
        self.periodic_y = periodic_y
        self.t_drain = t_drain


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


def make_2d_realization(rp: RunParameters, bf: BlobFactory):
    model = Model(
        Nx=rp.nx,
        Ny=rp.ny,
        Lx=rp.lx,
        Ly=rp.ly,
        dt=rp.dt,
        T=rp.T,
        num_blobs=rp.num_blobs,
        blob_shape=BlobShapeImpl(),
        periodic_y=rp.periodic_y,
        t_drain=rp.t_drain,
        blob_factory=bf,
        verbose=False,
        t_init=0,
    )
    return model.make_realization(speed_up=True, error=1e-10)

from typing import Union, List

import numpy as np
from nptyping import NDArray
import xarray as xr
from dataclasses import dataclass
from typing import Dict, Tuple

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


def run_parameters(rp: RunParameters, bf: BlobFactory):
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

    rp = RunParameters(T=T, lx=Lx, ly=Ly, nx=nx, ny=ny, dt=dt)
    bf = DeterministicBlobFactory(blobs)

    return run_parameters(rp, bf)


@dataclass
class BlobParameters:
    """
    Class describing the output blob parameters obtained with two-dimensional conditional averaging (2DCA)
    for GPI-APD data analysis. Stores velocity, area, shape, lifetime, and spatial scale parameters,
    with methods for validation, derived quantities, and data export.

    vx_c, vy_c and area_c are obtained with contouring methods.
    lx_f, ly_f, theta_f are obtained with a Gaussian fit.
    vx_tde, vy_tde are obtained with a 3-point TDE method.
    taud_psd and lambda_psd are obtained from fitting power spectral density.

    Attributes:
        vx_c (float): Center-of-mass velocity in x-direction (pixels/time step or m/s).
        vy_c (float): Center-of-mass velocity in y-direction (pixels/time step or m/s).
        area_c (float): Blob area (pixels² or m²).
        vx_tde (float): Velocity from time-dependent ellipse fitting in x-direction.
        vy_tde (float): Velocity from time-dependent ellipse fitting in y-direction.
        lx_f (float): Semi-major axis from Gaussian fit (pixels or m).
        ly_f (float): Semi-minor axis from Gaussian fit (pixels or m).
        theta_f (float): Ellipse rotation angle from Gaussian fit (radians).
        taud_psd (float): Blob lifetime from PSD analysis (time steps or ps).
        lambda_psd (float): Characteristic spatial scale from PSD analysis (pixels or m).
    """

    vx_c: float
    vy_c: float
    area_c: float
    vx_tde: float
    vy_tde: float
    lx_f: float
    ly_f: float
    theta_f: float
    taud_psd: float
    lambda_psd: float

    @property
    def velocity_c(self) -> Tuple[float, float]:
        """Return center-of-mass velocity vector (vx_c, vy_c)."""
        return (self.vx_c, self.vy_c)

    @property
    def velocity_tde(self) -> Tuple[float, float]:
        """Return time-dependent ellipse velocity vector (vx_tde, vy_tde)."""
        return (self.vx_tde, self.vy_tde)

    @property
    def total_velocity_c(self) -> float:
        """Compute magnitude of center-of-mass velocity."""
        return np.sqrt(self.vx_c**2 + self.vy_c**2)

    @property
    def total_velocity_tde(self) -> float:
        """Compute magnitude of time-dependent ellipse velocity."""
        return np.sqrt(self.vx_tde**2 + self.vy_tde**2)

    @property
    def aspect_ratio(self) -> float:
        """Compute aspect ratio of the fitted ellipse (lx_f / ly_f)."""
        return self.lx_f / self.ly_f if self.ly_f > 0 else np.inf

    @property
    def eccentricity(self) -> float:
        """Compute eccentricity of the fitted ellipse."""
        if self.lx_f >= self.ly_f:
            a, b = self.lx_f, self.ly_f
        else:
            a, b = self.ly_f, self.lx_f
        return np.sqrt(1 - (b / a) ** 2) if a > b else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to a dictionary for serialization."""
        return {
            "vx_c": self.vx_c,
            "vy_c": self.vy_c,
            "area_c": self.area_c,
            "vx_tde": self.vx_tde,
            "vy_tde": self.vy_tde,
            "lx_f": self.lx_f,
            "ly_f": self.ly_f,
            "theta_f": self.theta_f,
            "taud_psd": self.taud_psd,
            "lambda_psd": self.lambda_psd,
        }

    def __str__(self) -> str:
        """String representation of the blob parameters."""
        return (
            f"BlobParameters(vx_c={self.vx_c:.2f}, vy_c={self.vy_c:.2f}, "
            f"area_c={self.area_c:.2f}, vx_tde={self.vx_tde:.2f}, vy_tde={self.vy_tde:.2f}, "
            f"lx_f={self.lx_f:.2f}, ly_f={self.ly_f:.2f}, theta_f={self.theta_f:.2f}, "
            f"taud_psd={self.taud_psd:.2f}, lambda_psd={self.lambda_psd:.2f})"
        )

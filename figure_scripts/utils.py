from typing import Union, List

import numpy as np
from nptyping import NDArray
import xarray as xr

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
        prop_shape_parameters={"lam": 0.5},
        perp_shape_parameters={"lam": 0.5},
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
    vx,
    vy,
    lx,
    ly,
    theta,
    bs,
    blob_getter="deterministic",
    periodic_y=True,
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
        blob_shape=BlobShapeImpl(),
        periodic_y=periodic_y,  # Set to true so that high vertical velocities makes sense
        t_drain=1e10,
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


def make_2d_realization_test(
    Lx,
    Ly,
    T,
    nx,
    ny,
    dt,
    num_blobs,
    vx,
    vy,
    lx,
    ly,
    theta,
    bs,
    blob_getter="deterministic",
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
        blob_shape=BlobShapeImpl(),
        periodic_y=False,
        t_drain=1e10,
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


def get_average(shot, refx, refy, method_parameters):
    import os
    import imaging_methods as im

    file_name = os.path.join(
        "../density_scan/averages", f"average_ds_{shot}_{refx}{refy}.nc"
    )
    if not os.path.exists(file_name):
        print(f"File does not exist {file_name}")

        manager = im.GPIDataAccessor(
            "/home/sosno/Git/experimental_database/plasma_discharges.json"
        )
        ds = manager.read_shot_data(shot, preprocessed=True, data_folder="../data")
        events, average_ds = im.find_events_and_2dca(
            ds,
            refx,
            refy,
            threshold=method_parameters["2dca"]["threshold"],
            check_max=method_parameters["2dca"]["check_max"],
            window_size=method_parameters["2dca"]["window"],
            single_counting=method_parameters["2dca"]["single_counting"],
        )
        average_ds.to_netcdf(file_name)
        return average_ds
    average_ds = xr.open_dataset(file_name)
    if len(average_ds.data_vars) == 0:
        return None
    refx_ds, refy_ds = average_ds["refx"].item(), average_ds["refy"].item()
    assert refx == refx_ds and refy == refy_ds
    return average_ds

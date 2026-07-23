import numpy as np
from blobmodel import BlobShapeEnum, BlobShapeImpl, Geometry, Model
from imaging_methods import *

np.random.seed(42)

Lx = 10
Ly = 10
nx = 16
ny = 16
dt = 0.1
theta = 0
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

refx, refy = 5, 5


def make_deterministic_realization(blobs):
    geometry = Geometry(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly, dt=dt, T=10)
    model = Model.from_blobs(blobs, geometry=geometry)
    return model.make_realization(layout="imaging")


def test_com_and_velocities():
    vx, vy = 1, 0
    blobs = [
        get_blob(
            amplitude=1,
            vx=vx,
            vy=vy,
            posx=0,
            posy=5,
            lx=1,
            ly=1,
            t_init=0,
            bs=bs,
            theta=theta,
        )
    ]
    ds = make_deterministic_realization(blobs)
    ce = get_contour_evolution(ds.frames, 0.3)
    assert np.abs(ce.centroid.sel(time=5).values[0] - 5) < 0.1
    assert np.abs(ce.centroid.sel(time=4).values[0] - 4) < 0.1
    assert np.abs(ce.centroid.sel(time=6).values[0] - 6) < 0.1
    assert np.abs(np.max(ce.centroid.values[:, 1] - 5)) < 0.1

    assert np.abs(ce.position.sel(time=5).values[0] - 5) < 0.1
    assert np.abs(ce.position.sel(time=4).values[0] - 4) < 0.1
    assert np.abs(ce.position.sel(time=6).values[0] - 6) < 0.1
    assert np.abs(np.max(ce.position.values[:, 1] - 5)) < 0.1

    velocities = get_velocity_from_position(ce.centroid)
    vxs = velocities.sel(time=slice(3, 7)).values[:, 0]
    vys = velocities.sel(time=slice(3, 7)).values[:, 1]
    assert np.all(
        np.abs(vxs - 1) < 0.5
    )  # Due to pixel locking the velocity array varies a lot
    assert np.abs(np.mean(vxs) - 1) < 0.1
    assert np.all(np.abs(vys) < 0.1)


def test_largest_contour():
    vx, vy = 1, 0
    blobs = [
        get_blob(
            amplitude=1,
            vx=vx,
            vy=vy,
            posx=0,
            posy=3,
            lx=1,
            ly=1,
            t_init=0,
            bs=bs,
            theta=theta,
        ),
        get_blob(
            amplitude=0.5,
            vx=vx,
            vy=vy,
            posx=0,
            posy=7,
            lx=0.5,
            ly=0.5,
            t_init=0,
            bs=bs,
            theta=theta,
        ),
    ]
    ds = make_deterministic_realization(blobs)
    ce = get_contour_evolution(ds.frames, 0.3)
    assert np.abs(ce.centroid.sel(time=5).values[0] - 5) < 0.1
    assert np.abs(ce.centroid.sel(time=4).values[0] - 4) < 0.1
    assert np.abs(ce.centroid.sel(time=6).values[0] - 6) < 0.1
    assert np.abs(np.max(ce.centroid.values[:, 1] - 3)) < 0.1

import imaging_methods as im
from test_utils import (
    make_2d_realization,
    get_blob,
    DeterministicBlobFactory,
    Model,
)
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import pytest

Lx = 10
Ly = 10
nx = 1
ny = 1
dt = 0.01
theta = 0
bs = BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.exp)

refx, refy = 0, 0


@pytest.mark.parametrize(
    "sos", [im.SecondOrderStatistic.PSD, im.SecondOrderStatistic.ACF]
)
@pytest.mark.parametrize("size", [0.1, 1, 10])
def test_size(sos, size):
    ds = make_2d_realization(
        Lx,
        Ly,
        10000,
        nx,
        ny,
        dt,
        1000,
        vx=1,
        vy=0,
        lx=size,
        ly=size,
        theta=theta,
        bs=bs,
    )

    cutoff = None
    if sos == im.SecondOrderStatistic.ACF:
        cutoff = 10 * size

    taud, _ = im.DurationTimeEstimator(
        sos, im.Analytics.TwoSided
    ).estimate_duration_time(
        ds.isel(x=refx, y=refy).frames.values,
        dt,
        cutoff=cutoff,
        nperseg=1000,
    )

    expected = size
    rel_error = (taud - expected) / expected
    assert np.abs(rel_error < 0.2), "Error too large"


@pytest.mark.parametrize(
    "sos", [im.SecondOrderStatistic.PSD, im.SecondOrderStatistic.ACF]
)
@pytest.mark.parametrize("vx", [0.1, 1, 10])
def test_vel(sos, vx):
    ds = make_2d_realization(
        Lx, Ly, 10000, nx, ny, dt, 1000, vx=vx, vy=0, lx=1, ly=1, theta=theta, bs=bs
    )

    cutoff = None
    if sos == im.SecondOrderStatistic.ACF:
        cutoff = 10 / vx

    taud, _ = im.DurationTimeEstimator(
        sos, im.Analytics.TwoSided
    ).estimate_duration_time(
        ds.frames.isel(x=refx, y=refy).values,
        dt,
        cutoff=cutoff,
        nperseg=1000,
    )

    expected = 1 / vx
    rel_error = (taud - expected) / expected
    assert np.abs(rel_error < 0.2), "Error too large"


def test_uniform_velocity_distribution():
    T = 1000
    blobs = [
        get_blob(
            amplitude=1,
            vx=np.random.uniform(0.5, 1.5),
            vy=0,
            posx=np.random.uniform(0, Lx),
            posy=np.random.uniform(0, Ly),
            lx=1,
            ly=1,
            t_init=np.random.uniform(0, T),
            bs=bs,
        )
        for _ in range(100)
    ]

    bf = DeterministicBlobFactory(blobs)

    model = Model(
        Nx=nx,
        Ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        T=T,
        num_blobs=100,
        blob_shape=BlobShapeImpl(),
        periodic_y=False,
        t_drain=1e10,
        blob_factory=bf,
        verbose=False,
        t_init=0,
    )
    ds = model.make_realization(speed_up=True, error=1e-10)
    data_series = ds.n.isel(x=refx, y=refy).values

    taud, _ = im.DurationTimeEstimator(
        im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
    ).estimate_duration_time(
        data_series,
        dt,
        cutoff=None,
        nperseg=1000,
    )

    expected = 1
    rel_error = (taud - expected) / expected
    assert np.abs(rel_error < 0.2), "Error too large"

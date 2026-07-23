import numpy as np

from blobmodel import Blob, Geometry, Model

__all__ = [
    "get_blob",
    "make_2d_realization",
]


def get_blob(
    amplitude,
    vx,
    vy,
    posx,
    posy,
    lx,
    ly,
    t_init,
    theta=None,
    bs=None,
    taup=np.inf,
):
    """
    Convenience wrapper around blobmodel.Blob using imaging-methods naming
    (vx/vy, lx/ly, posx/posy, taup). Pulse shapes are symmetric (lam=0.5).

    With theta=None (default) the blob is aligned with its velocity; an
    explicit theta always wins.
    """
    return Blob(
        blob_shape=bs,
        amplitude=amplitude,
        width_p=lx,
        width_s=ly,
        v_x=vx,
        v_y=vy,
        pos_x0=posx,
        pos_y0=posy,
        t_init=t_init,
        t_drain=taup,
        shape_parameters_p={"lam": 0.5},
        shape_parameters_s={"lam": 0.5},
        blob_alignment=True,
        theta=theta,
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
    theta=None,
    bs=None,
    blob_getter=None,
    periodic_y=True,
):
    """
    Make a 2D blobmodel realization in the cmod_functions imaging format
    (frames(y, x, time) with R/Z coordinate grids).

    If blob_getter is None, num_blobs blobs are drawn with exponentially
    distributed amplitudes, uniform posy and t_init, and the given vx, vy,
    lx, ly, theta (amplitudes and positions come from the global np.random
    state). Otherwise blob_getter is a zero-argument callable returning a
    Blob, called num_blobs times.
    """
    if blob_getter is None:
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

    geometry = Geometry(
        Nx=nx,
        Ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        T=T,
        periodic_y=periodic_y,  # Set to true so that high vertical velocities makes sense
    )
    model = Model.from_blobs(blobs, geometry=geometry)
    return model.make_realization(layout="imaging")

from typing import Union, List
from nptyping import NDArray

from .utils import get_dr, get_dt, smooth_da
from .cond_av import find_events_and_2dca
from .contours import get_contour_evolution
from .velocity_estimates import *
from .maximum_trajectory import *
from .method_parameters import *
import velocity_estimation as ve

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


def get_averaged_velocity(
    average_ds, variable, method_parameters, position_method="contouring"
):
    if position_method == "contouring":
        position_da = get_contour_evolution(
            average_ds[variable],
            method_parameters.contouring.threshold_factor,
            max_displacement_threshold=None,
        ).center_of_mass
    elif position_method == "max":
        position_da = compute_maximum_trajectory_da(
            average_ds, variable, method="parabolic"
        )
    else:
        raise NotImplementedError

    position_da, start, end = smooth_da(
        position_da, method_parameters.position_filter, return_start_end=True
    )

    mask = get_combined_mask(
        average_ds.isel(time=slice(start, end)),
        variable,
        position_da,
        method_parameters.position_filter,
    )

    v, w = get_averaged_velocity_from_position(
        position_da=position_da,
        mask=mask,
    )
    return v, w


def estimate_velocities_synthetic_ds(ds, method_parameters: MethodParameters):
    """
    Estimates blob velocities from a given synthetic data dataset and parameters given in method_parameters.
    """
    dt = get_dt(ds)

    events, average_ds = find_events_and_2dca(ds, method_parameters.two_dca)

    v_2dca, w_2dca = get_averaged_velocity(
        average_ds, "cond_av", method_parameters, position_method="contouring"
    )
    v_2dcc, w_2dcc = get_averaged_velocity(
        average_ds, "cross_corr", method_parameters, position_method="contouring"
    )

    v_2dca_max, w_2dca_max = get_averaged_velocity(
        average_ds, "cond_av", method_parameters, position_method="max"
    )
    v_2dcc_max, w_2dcc_max = get_averaged_velocity(
        average_ds, "cross_corr", method_parameters, position_method="max"
    )

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters.two_dca.window * dt
    eo.cc_options.minimum_cc_value = 0
    eo.cc_options.running_mean = False
    try:
        pd = ve.estimate_velocities_for_pixel(
            method_parameters.two_dca.refx,
            method_parameters.two_dca.refy,
            ve.CModImagingDataInterface(ds),
        )
        vx, vy = pd.vx, pd.vy
    except ValueError:
        print("LOL")
        vx, vy = np.nan, np.nan

    return (
        v_2dca,
        w_2dca,
        v_2dcc,
        w_2dcc,
        v_2dca_max,
        w_2dca_max,
        v_2dcc_max,
        w_2dcc_max,
        vx,
        vy,
    )

from blobmodel import BlobShapeEnum
from imaging_methods import *

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)


MAKE_PLOTS = True
T = 5000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 500

alpha = np.pi / 8
vx_input = np.cos(alpha)
vy_input = np.sin(alpha)


def get_synthetic_data(run_norm_radius):
    lx_input = 1 / 2
    ly_input = 2
    theta_input = np.pi / 4

    ds = make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_input,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = run_norm_ds(ds, run_norm_radius)
    return ds


ds = get_synthetic_data(1000)


def test_synthetic_data():
    method_parameters = get_default_synthetic_method_params()

    tdca_params = method_parameters.two_dca
    events, average_ds = find_events_and_2dca(ds, tdca_params)

    position_da = get_contour_evolution(
        average_ds.cond_av,
        method_parameters.contouring.threshold_factor,
        max_displacement_threshold=None,
    ).center_of_mass

    R, Z = (
        average_ds.R.isel(x=tdca_params.refx, y=tdca_params.refy).item(),
        average_ds.Z.isel(x=tdca_params.refx, y=tdca_params.refy).item(),
    )
    times = position_da.time.values
    real_position = np.array([R + vx_input * times, Z + vy_input * times]).T

    max_error_2dca = np.max(np.abs(real_position - position_da))

    print("Error 2DCA: {:.4f}".format(max_error_2dca))

    assert max_error_2dca < 1

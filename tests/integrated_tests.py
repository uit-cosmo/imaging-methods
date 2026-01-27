"""
Tests the full analysis pipeline in several scenarios. Some of the tests fail sometimes as the methods are not fully
developed yet.
"""

from test_utils import *
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)


MAKE_PLOTS = True
T = 1000
Lx = 8
Ly = 8
nx = 16
ny = 16
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 1000

method_parameters = im.get_default_synthetic_method_params()

figures_dir = "integrated_tests_figures"


def test_case_a():
    """
    Base case
    """
    vx_input = 1
    vy_intput = 0
    lx_input = 1
    ly_input = 1
    theta_input = 0

    ds = make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    bp = full_analysis(
        ds, method_parameters, "a", do_plots=MAKE_PLOTS, variable="cross_corr"
    )
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.05, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.05, "Wrong contour y velocity"

    assert np.abs(bp.vx_2dca_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_2dca_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_b():
    """
    Elongated pulses
    """
    vx_input, vy_input = 1, 0
    aspect_ratio = 4
    lx_input = 1 / np.sqrt(aspect_ratio)
    ly_input = np.sqrt(aspect_ratio)
    theta_input = np.pi / 4

    ds = make_2d_realization(
        Lx,
        Ly,
        T,
        16,
        16,
        dt,
        K,
        vx=vx_input,
        vy=vy_input,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    bp = full_analysis(
        ds, method_parameters, "b", do_plots=MAKE_PLOTS, variable="cross_corr"
    )
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.05, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_input) < 0.05, "Wrong contour y velocity"

    # No TDE velocity asserts as TDE will be wrong due to barberpole effects.

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_c():
    """
    Noise
    """
    vx_input = 1
    vy_intput = 0
    aspect_ratio = 1
    lx_input = np.sqrt(aspect_ratio)
    ly_input = 1 / np.sqrt(aspect_ratio)
    theta_input = 0

    ds = make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)

    sigma = 1
    ds = ds.assign(frames=ds["frames"] + sigma * np.random.random(ds.frames.shape))
    bp = full_analysis(ds, method_parameters, "c", do_plots=MAKE_PLOTS)
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    assert np.abs(bp.vx_2dca_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_2dca_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_e():
    """
    Low spatial resolution
    """
    vx_input = 1
    vy_intput = 0
    lx_input = 1
    ly_input = 1
    theta_input = 0
    method_parameters_e = method_parameters
    method_parameters_e.two_dca.refx = 4
    method_parameters_e.two_dca.refy = 4

    ds = make_2d_realization(
        Lx,
        Ly,
        T,
        8,
        8,
        dt,
        K,
        vx=vx_input,
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    bp = full_analysis(ds, method_parameters, "e")
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    assert np.abs(bp.vx_2dca_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_2dca_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_f():
    """
    Distribution of velocities
    """

    lx_input = 1
    ly_input = 1
    theta_input = 0
    vx_input = 1
    vy_input = 0

    def blob_getter():
        u1 = np.random.uniform(0.75, 1.25)
        u2 = np.random.uniform(-0.25, 0.25)
        return get_blob(
            amplitude=np.random.exponential(),
            vx=u1,
            vy=u2,
            posx=0,
            posy=np.random.uniform(0, Ly),
            lx=1,
            ly=1,
            t_init=np.random.uniform(0, T),
            bs=bs,
            theta=0,
        )

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
        blob_getter=blob_getter,
    )
    ds = im.run_norm_ds(ds, method_parameters.preprocessing.radius)
    bp = full_analysis(ds, method_parameters, "a", do_plots=MAKE_PLOTS)
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.05, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_input) < 0.05, "Wrong contour y velocity"

    assert np.abs(bp.vx_2dca_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_2dca_tde - vy_input) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"

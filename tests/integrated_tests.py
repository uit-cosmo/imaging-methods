from test_utils import *
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np

T = 500
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.025
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = T * Ly

# Method parameters
method_parameters = {
    "preprocessing": {"radius": 1000},
    "2dca": {
        "refx": 4,
        "refy": 4,
        "threshold": 2,
        "window": 150,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 3},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

figures_dir = "integrated_tests_figures"


def test_case_a():
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
    ds = ph.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
    bp = full_analysis(ds, method_parameters, "a")
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    assert np.abs(bp.vx_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_b():
    vx_input = 1
    vy_intput = 0
    aspect_ratio = 3
    lx_input = np.sqrt(aspect_ratio)
    ly_input = 1 / np.sqrt(aspect_ratio)
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
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = ph.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
    bp = full_analysis(ds, method_parameters, "b")
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    # No TDE velocity asserts as TDE will be wrong due to barberpole effects.

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_c():
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
    ds = ph.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])

    sigma = 1
    ds = ds.assign(frames=ds["frames"] + sigma * np.random.random(ds.frames.shape))
    bp = full_analysis(ds, method_parameters, "c")
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    assert np.abs(bp.vx_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_d():
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
        K * 50,
        vx=vx_input,
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
    )
    ds = ph.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])

    sigma = 1
    ds = ds.assign(frames=ds["frames"] + sigma * np.random.random(ds.frames.shape))
    bp = full_analysis(ds, method_parameters, "d")
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    assert np.abs(bp.vx_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"


def test_case_e():
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
        4,
        4,
        dt,
        K * 50,
        vx=vx_input,
        vy=vy_intput,
        lx=lx_input,
        ly=ly_input,
        theta=theta_input,
        bs=bs,
        )
    method_parameters_e = method_parameters
    method_parameters_e["2dca"]["refx"] = 2
    method_parameters_e["2dca"]["refy"] = 2

    ds = ph.run_norm_ds(ds, method_parameters_e["preprocessing"]["radius"])

    bp = full_analysis(ds, method_parameters_e, "e")
    print(bp)

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour x velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour y velocity"

    assert np.abs(bp.vx_tde - vx_input) < 0.2, "Wrong TDE x velocity"
    assert np.abs(bp.vy_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"

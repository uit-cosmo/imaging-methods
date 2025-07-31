import phantom as ph
from test_utils import *
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import velocity_estimation as ve

T = 500
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.025
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = T * Ly

refx, refy = 4, 4

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

    assert np.abs(bp.vx_c - vx_input) < 0.2, "Wrong contour velocity"
    assert np.abs(bp.vy_c - vy_intput) < 0.2, "Wrong contour velocity"

    assert np.abs(bp.vx_tde - vx_input) < 0.2, "Wrong TDE velocity"
    assert np.abs(bp.vy_tde - vy_intput) < 0.2, "Wrong TDE velocity"

    assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
    assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"

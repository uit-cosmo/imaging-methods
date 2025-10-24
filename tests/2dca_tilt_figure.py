import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
from test_utils import *
import cosmoplots as cp

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)


T = 10000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 1000

# Method parameters
method_parameters = {
    "preprocessing": {"radius": 1000},
    "2dca": {
        "refx": 4,
        "refy": 4,
        "threshold": 2,
        "window": 300,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 10},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

figures_dir = "integrated_tests_figures"

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
ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])
bp = full_analysis(ds, method_parameters, "a")
print(bp)

assert np.abs(bp.vx_c - vx_input) < 0.05, "Wrong contour x velocity"
assert np.abs(bp.vy_c - vy_intput) < 0.05, "Wrong contour y velocity"

assert np.abs(bp.vx_2dca_tde - vx_input) < 0.2, "Wrong TDE x velocity"
assert np.abs(bp.vy_2dca_tde - vy_intput) < 0.2, "Wrong TDE y velocity"

assert np.abs(bp.taud_psd - 1) < 0.5, "Wrong duration time"
assert np.abs(bp.theta_f - theta_input) < 0.1, "Wrong tilt angle"
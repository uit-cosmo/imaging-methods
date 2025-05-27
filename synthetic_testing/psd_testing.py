from phantom.show_data import *
from phantom.cond_av import *
from phantom.contours import *
from phantom.parameter_estimation import *
import phantom as ph
from realizations import make_2d_realization
from blobmodel import BlobShapeEnum, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np
import cosmoplots as cp

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

from synthetic_testing.realizations import BlobParameters

num_blobs = 500
T = 2000
Lx = 10
Ly = 10
aspect_ratio = 1  # lx/ly
lx = np.sqrt(aspect_ratio)
ly = 1 / np.sqrt(aspect_ratio)
nx = 8
ny = 8
dt = 0.1
vx = 1
vy = 0
theta = -np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian)
ds = make_2d_realization(Lx, Ly, T, nx, ny, dt, num_blobs, vx, vy, lx, ly, theta, bs)

refx, refy = 4, 4
fig, ax = cp.figure_multiple_rows_columns(1, 1)
ax = ax[0]
taud, lam = ph.fit_psd(
    ds.frames.isel(x=refx, y=refy).values,
    get_dt(ds),
    nperseg=10**3,
    ax=ax,
    cutoff_freq=1e6,
    relative=False,
)
plt.show()

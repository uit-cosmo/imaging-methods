from synthetic_data import *
from phantom.utils import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import numpy as np


def get_blob(vx, vy, posx, posy, lx, ly, t_init, theta, bs=BlobShapeImpl()):
    return Blob(
        1,
        bs,
        amplitude=1,
        width_prop=lx,
        width_perp=ly,
        v_x=vx,
        v_y=vy,
        pos_x=posx,
        pos_y=posy,
        t_init=t_init,
        t_drain=1e100,
        theta=theta,
        blob_alignment=True if theta == 0 else False,
    )


num_blobs = 1000
T = 2000
Lx = 3
Ly = 3
lx = 0.5
ly = 1.5
nx = 5
ny = 5
vx = 1
vy = -1
theta = -np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

blobs = [
    get_blob(
        vx=vx,
        vy=vy,
        posx=np.random.uniform(0, Lx),
        posy=np.random.uniform(0, Ly),
        lx=lx,
        ly=ly,
        t_init=np.random.uniform(0, T),
        bs=bs,
        theta=theta,
    )
    for _ in range(num_blobs)
]

rp = RunParameters(T=T, lx=Lx, ly=Ly, nx=nx, ny=ny)
bf = DeterministicBlobFactory(blobs)

ds = make_2d_realization(rp, bf)

refx, refy = 2, 2

events = find_events(ds, refx, refy, threshold=0.2, check_max=2)
average = compute_average_event(events)

fig, ax = plt.subplots()
rx, ry = average.R.isel(x=refx, y=refy).item(), average.Z.isel(x=refx, y=refy).item()
R_min, R_max = average.R.min().item(), average.R.max().item()
Z_min, Z_max = average.Z.min().item(), average.Z.max().item()

average_blob = average.sel(time=0).frames.values

im = ax.imshow(
    average_blob,
    origin="lower",
    interpolation="spline16",
    extent=(R_min, R_max, Z_min, Z_max),
)
ax.scatter(rx, ry)


def model(params):
    """Objective function with regularization"""
    blob = rotated_blob(params, rx, ry, average.R.values, average.Z.values)
    diff = blob - average_blob

    # Add regularization to prevent lx/ly from collapsing
    reg = 0.01 * (1 / lx**2 + 1 / ly**2)
    return np.sum(diff**2) + reg


# Initial guesses for lx, ly, and t
# Rough estimation
bounds = [
    (0, 5),  # lx: 0 to 5
    (0, 5),  # ly: 0 to 5
    (-np.pi / 4, np.pi / 4),  # t: 0 to 2π
]

result = differential_evolution(
    model,
    bounds,
    seed=42,  # Optional: for reproducibility
    popsize=15,  # Optional: population size multiplier
    maxiter=1000,  # Optional: maximum number of iterations
)

alphas = np.linspace(0, 2 * np.pi, 200)
elipsx, elipsy = zip(*[ellipse_parameters(result.x, rx, ry, a) for a in alphas])
ax.plot(elipsx, elipsy)

print(result.x)

plt.show()
print("LOL")

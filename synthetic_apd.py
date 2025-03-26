from synthetic_data import *
from phantom.show_data import show_movie
from phantom.utils import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt

case = 3


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


num_blobs = 200
T = 500
Lx = 4
Ly = 8
lx = 0.5
ly = 1.5
nx = 4
ny = 8
vx = 1
vy = 0
theta = -np.pi / 4
bs = BlobShapeImpl(BlobShapeEnum.rect, BlobShapeEnum.rect)

blobs = [
    get_blob(
        vx=vx,
        vy=vy,
        posx=np.random.uniform(0, Lx),
        posy=Ly,
        lx=lx,
        ly=ly,
        t_init=np.random.uniform(0, T),
        theta=theta,
    )
    for _ in range(num_blobs)
]
# blobs = [get_blob(vx=vx, vy=vy, posx=0, posy=np.random.uniform(0, Ly), lx=lx, ly=ly, t_init=np.random.uniform(0, T), theta=theta) for _ in range(num_blobs)]

rp = RunParameters(T=T, lx=Lx, ly=Ly, nx=nx, ny=ny)
bf = DeterministicBlobFactory(blobs)

ds = make_2d_realization(rp, bf)

show_mo = True

if show_mo:
    show_movie(
        ds.sel(time=slice(T / 2, min(T / 2 + 10, T))),
        variable="frames",
        gif_name="synthetic{}.gif".format(case),
        fps=60,
        interpolation="spline16",
        lims=(0, 1 / np.pi),
    )

refx, refy = 2, 4
delta = 10

fig, ax = plt.subplots(5, 4, sharex=True, figsize=(10, 20))
plot_ccfs_grid(
    ds,
    ax,
    refx,
    refy,
    [6, 5, 4, 3, 2],
    [0, 1, 2, 3],
    delta=delta,
    plot_tau_M=True,
    vx=vx,
    vy=vy,
    lx=lx,
    ly=ly,
    theta=theta,
)

name = "grid_ccf{}.eps".format(case)
plt.savefig(name, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(4, 5, figsize=(40, 50))
yvals = [6, 5, 4, 3, 2]
xvals = [0, 1, 2, 3]

for x in range(4):
    for y in range(5):
        lx, ly, t = plot_2d_ccf(ds, xvals[x], yvals[y], delta, ax[x, y])
        ax[x, y].set_title(r"lx = {:.2f}, ly = {:.2f}, a = {:.2f}".format(lx, ly, t))

plt.savefig("2d_ccf_{}".format(case), bbox_inches="tight")
plt.show()

eo = ve.EstimationOptions(
    cc_options=ve.CCOptions(cc_window=delta, minimum_cc_value=0.5, interpolate=True)
)
pd = ve.estimate_velocities_for_pixel(
    refx, refy, ve.CModImagingDataInterface(ds), estimation_options=eo
)
vx, vy = pd.vx, pd.vy
print("Velcities estimated with 3TDE v={:.2f}, w={:.2f}".format(vx, vy))

u2 = vx**2 + vy**2
taux = (vx * 1) / u2
tauy = (vy * 1) / u2
lx, ly, t = plot_2d_ccf(ds, refx, refy, delta, None)


def error_tau(params):
    vx, vy = params
    taux_given = get_taumax(vx, vy, 1, 0, lx, ly, t)  # Calculate tau_x
    tauy_given = get_taumax(vx, vy, 0, 1, lx, ly, t)  # Calculate tau_y
    return (taux_given - taux) ** 2 + (tauy_given - tauy) ** 2


bonds = [(-10, 10), (-10, 10)]
result_u = differential_evolution(
    error_tau,
    bonds,
    seed=42,  # Optional: for reproducibility
    popsize=15,  # Optional: population size multiplier
    maxiter=1000,  # Optional: maximum number of iterations
)

print(
    "Corrected velocities assuming lx={:.2f}, ly={:.2f} and t={:.2f} are v={:.2f} and w={:.2f}".format(
        lx, ly, t, result_u.x[0], result_u.x[1]
    )
)

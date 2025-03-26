import numpy as np

from phantom.show_data import *
from phantom.utils import *
import fppanalysis as fpp
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import velocity_estimation as ve

shot = 1140613026

ds = xr.open_dataset("ds_short.nc")

times = ds["time"].values
refx, refy = 7, 5


delta = 1e-5
eo = ve.EstimationOptions(cc_options=ve.CCOptions(cc_window=delta, minimum_cc_value=0.5, interpolate=True))
tded = ve.TDEDelegator(ve.TDEMethod.CC, ve.CCOptions(cc_window=delta, minimum_cc_value=0.2, interpolate=True), cache=False)

fig, ax = plt.subplots(5, 4, sharex=True, figsize=(10, 20))
plot_ccfs_grid(ds, ax, refx, refy, [7, 6, 5, 4, 3], [5, 6, 7, 8], delta=delta)

ccf = True
name = "grid_ccf{}{}.eps".format(refx, refy) if ccf else "grid_ca{}{}.eps".format(refx, refy)
plt.savefig(name, bbox_inches="tight")
plt.show()

ds_short = ds.isel(x=slice(-4, None))
refx = refx - 5


# Define the function to fit
fig, ax = plt.subplots(3, 5, figsize=(30, 50))
yvals = [7, 5, 4, 3, 1]
xvals = [1, 2, 3]

for x in range(3):
    for y in range(5):
        lx, ly, t = plot_2d_ccf(ds, xvals[x], yvals[y], delta, ax[x, y])
        ax[x, y].set_title(r"lx = {:.2f}, ly = {:.2f}, a = {:.2f}".format(lx, ly, t))

plt.savefig("2d_ccf_{}{}".format(refx+5, refy), bbox_inches="tight")
plt.show()

ds_corr = get_2d_corr(ds, refx, refy, delta)

pd = ve.estimate_velocities_for_pixel(refx, refy, ve.CModImagingDataInterface(ds_short), estimation_options=eo)
vx, vy = pd.vx, pd.vy
print("Velcities estimated with 3TDE v={:.2f}, w={:.2f}".format(vx, vy))
deltax = ds_corr.R.isel(x=refx, y=refy).values - ds_corr.R.isel(x=refx-1, y=refy).values
deltay = ds_corr.Z.isel(x=refx, y=refy).values - ds_corr.Z.isel(x=refx, y=refy-1).values

u2 = vx ** 2 + vy ** 2
taux = (vx * deltax) / u2
tauy = (vy * deltay) / u2


lx, ly, t = plot_2d_ccf(ds, refx, refy, delta, None)

def error_tau(params):
    vx, vy = params
    taux_given = get_taumax(vx, vy, deltax, 0, lx, ly, t)  # Calculate tau_x
    tauy_given = get_taumax(vx, vy, 0, deltay, lx, ly, t)  # Calculate tau_y
    return (taux_given - taux) ** 2 + (tauy_given - tauy) ** 2


bonds = [(-5e6, 5e6), (-5e6, 5e6)]
result_u = differential_evolution(
    error_tau,
    bonds,
    seed=42,  # Optional: for reproducibility
    popsize=15,  # Optional: population size multiplier
    maxiter=1000  # Optional: maximum number of iterations
)


print("Corrected velocities assuming lx={:.2f}, ly={:.2f} and t={:.2f} are v={:.2f} and w={:.2f}".format(lx, ly, t, result_u.x[0], result_u.x[1]))



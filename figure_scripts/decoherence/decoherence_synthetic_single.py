import numpy as np

from decoherence_utils import *
import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

T = 1000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 1000

vx_input = 1
vy_intput = 0
lx_input = 1
ly_input = 1

alpha_max = np.pi / 4
vx_input = 2 * np.sin(alpha_max) / (2 * alpha_max) if alpha_max != 0 else 1
vy_input = 0


def blob_getter():
    alpha = np.random.uniform(-alpha_max, alpha_max)
    return get_blob(
        amplitude=np.random.exponential(),
        vx=1,#np.random.uniform(0.5, 1.5),
        vy=0,#np.random.uniform(-1, 1),
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
    vy=vy_intput,
    lx=lx_input,
    ly=ly_input,
    theta=0,
    bs=bs,
    blob_getter=blob_getter
)
ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])

tdca_params = method_parameters["2dca"]
events, average_ds = im.find_events_and_2dca(
    ds,
    tdca_params["refx"],
    tdca_params["refy"],
    threshold=tdca_params["threshold"],
    check_max=tdca_params["check_max"],
    window_size=tdca_params["window"],
    single_counting=tdca_params["single_counting"],
)

contour_ds = im.get_contour_evolution(
    average_ds.cond_av,
    method_parameters["contouring"]["threshold_factor"],
    max_displacement_threshold=None,
)

velocity_ds = im.get_contour_velocity(
    contour_ds.center_of_mass,
    method_parameters["contouring"]["com_smoothing"],
)

vx_c, vy_c, vx_cc_tde, vy_cc_tde, confidence, vx_2dca_tde, vy_2dca_tde, cond_repr = estimate_velocities(ds, method_parameters)

gif_file_name = "contours_decoherence_synthetic.gif"

plot_contours = False

if plot_contours:
    im.show_movie_with_contours(
        average_ds,
        contour_ds,
        apd_dataset=None,
        variable="cond_av",
        lims=(0, 3),
        gif_name=gif_file_name,
        interpolation="spline16",
        show=False,
    )

    im.show_movie_with_contours(
        average_ds,
        contour_ds,
        apd_dataset=None,
        variable="cond_repr",
        lims=(0, 1),
        gif_name="cond_repr.gif",
        interpolation="spline16",
        show=False,
    )

print("CC 3TDE velocities: {:.2f} {:.2f}".format(vx_cc_tde, vy_cc_tde))
print("2DCA 3TDE velocities: {:.2f} {:.2f}".format(vx_2dca_tde, vy_2dca_tde))
print("C velocities: {:.2f} {:.2f}".format(vx_c, vy_c))
print("Avg input velocities: {:.2f} {:.2f}".format(vx_input, vy_input))
print("Confidence: {:.2f}".format(confidence))
print("Cond. Repr.: {:.2f}".format(cond_repr))

tau_x_index = int(1 / vx_c / dt)
print("Cond repr at neighbour: {:.2f}".format(np.max(average_ds.cond_repr.isel(x=5, y=4).values)))

fig, ax = plt.subplots()

ax.plot(average_ds.time.values, average_ds.cond_av.isel(x=3, y=4).values)
ax.plot(average_ds.time.values, average_ds.cond_av.isel(x=4, y=4).values)
ax.plot(average_ds.time.values, average_ds.cond_av.isel(x=5, y=4).values)
ax.plot(average_ds.time.values, average_ds.cond_av.isel(x=4, y=3).values)
ax.plot(average_ds.time.values, average_ds.cond_av.isel(x=4, y=5).values)

fig, ax = plt.subplots()

ax.plot(average_ds.time.values, average_ds.cond_repr.isel(x=3, y=4).values)
ax.plot(average_ds.time.values, average_ds.cond_repr.isel(x=4, y=4).values)
ax.plot(average_ds.time.values, average_ds.cond_repr.isel(x=5, y=4).values)
ax.plot(average_ds.time.values, average_ds.cond_repr.isel(x=4, y=3).values)
ax.plot(average_ds.time.values, average_ds.cond_repr.isel(x=4, y=5).values)

fig, ax = plt.subplots()

ax.plot(average_ds.time.values, average_ds.cond_av.max(dim=["x", "y"]).values)
ax.plot(average_ds.time.values, average_ds.cond_repr.max(dim=["x", "y"]).values)

plt.show()

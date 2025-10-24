import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
from matplotlib import matplotlib_fname
from scipy import interpolate

import imaging_methods as im
import cosmoplots as cp
from collections import defaultdict
import matplotlib as mpl
import matplotlib.patches as mpatches

matplotlib_params = plt.rcParams
cp.set_rcparams_dynamo(matplotlib_params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(matplotlib_params)
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

results = im.ResultManager.from_json("results.json")
manager = im.GPIDataAccessor()
manager.load_from_json("plasma_discharges.json")

# shot = 1160616027
shot = 1140613026
# shot = 1120814031
ds = manager.read_shot_data(shot, data_folder="../data")

fig, ax = plt.subplots(1, 2, figsize=(2 * 3.3, 1 * 3.3), gridspec_kw={"wspace": 0.5})
im.plot_skewness_and_flatness(ds, shot, fig, ax)

ne = np.transpose(results.get_blob_param_array(shot, "number_events"))
vx_c = np.transpose(results.get_blob_param_array(shot, "vx_c"))
vy_c = np.transpose(results.get_blob_param_array(shot, "vy_c"))
vx_2dca_tde = np.transpose(results.get_blob_param_array(shot, "vx_2dca_tde"))
vy_2dca_tde = np.transpose(results.get_blob_param_array(shot, "vy_2dca_tde"))
vx_tde = np.transpose(results.get_blob_param_array(shot, "vx_tde"))
vy_tde = np.transpose(results.get_blob_param_array(shot, "vy_tde"))
lx = np.transpose(results.get_blob_param_array(shot, "lx_f"))
ly = np.transpose(results.get_blob_param_array(shot, "ly_f"))
theta = np.transpose(results.get_blob_param_array(shot, "theta_f"))


def plot_velocity_field(vx, vy, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    qiv = ax.quiver(
        ds.R.values,
        ds.Z.values,
        vx,
        vy,
        ne,
        scale_units="xy",
        angles="xy",
    )
    qk = ax.quiverkey(
        qiv,
        0.63,
        1.025,
        500,
        r"$500$ m\,/\,s",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 6},
        labelsep=0.02,
    )
    cbar = fig.colorbar(qiv, format="%.1f")
    cbar.ax.set_ylabel(r"\#Events", rotation=270, labelpad=13)
    im.add_limiter_and_lcfs(ds, ax)
    ax.set_title(title)
    ax.set_ylim((ds.Z[0, 0] - 0.25, ds.Z[-1, 0] + 0.25))
    ax.set_xlim((ds.R[0, 0] - 0.25, ds.R[0, -1] + 0.25))


def plot_ellipse_field(lx, ly, theta, title):
    fig, ax = plt.subplots(figsize=(3, 3))

    # Normalize ne for colormap
    norm = plt.Normalize(np.nanmin(ne), np.nanmax(ne))
    cmap = plt.get_cmap("viridis")

    # Scaling factor for ellipse sizes (adjust as needed)
    scale_factor = 20  # Adjust to make ellipses visible but not overlapping

    # Add ellipse for each pixel
    for i in range(10):
        for j in range(9):
            if (
                not np.isnan(lx[i, j])
                and not np.isnan(ly[i, j])
                and not np.isnan(theta[i, j])
            ):
                ellipse = mpatches.Ellipse(
                    xy=(ds.R.values[i, j], ds.Z.values[i, j]),  # Center of ellipse
                    width=2 * lx[i, j] * scale_factor,  # Full width (2 * semi-major)
                    height=2 * ly[i, j] * scale_factor,  # Full height (2 * semi-minor)
                    angle=theta[i, j] * 360 / (2 * np.pi),  # Rotation in degrees
                    facecolor=cmap(norm(ne[i, j])),  # Color by ne
                    edgecolor="none",  # No border for clarity
                    alpha=0.7,  # Slight transparency
                )
                ax.add_patch(ellipse)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, format="%.1f")
    cbar.ax.set_ylabel(r"\#Events", rotation=270, labelpad=13)

    # Add limiter and LCFS
    im.add_limiter_and_lcfs(ds, ax)

    # Set plot properties
    ax.set_title(title)
    ax.set_ylim((ds.Z[0, 0] - 0.25, ds.Z[-1, 0] + 0.25))
    ax.set_xlim((ds.R[0, 0] - 0.25, ds.R[0, -1] + 0.25))
    ax.set_xlabel("R")
    ax.set_ylabel("Z")

    return fig, ax


plot_velocity_field(vx_c, vy_c, r"$v_c$")
plt.savefig("vc_{}.pdf".format(shot), bbox_inches="tight")
plot_velocity_field(vx_2dca_tde, vy_2dca_tde, r"$v_\text{TDE}^\text{2dca}$")
plot_velocity_field(vx_tde, vy_tde, r"$v_\text{TDE}$")
plot_ellipse_field(lx, ly, theta, r"Blob fit")


plt.show()

print("LOL")

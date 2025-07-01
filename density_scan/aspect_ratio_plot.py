import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import phantom as ph
import cosmoplots as cp

suffixes = ["66", "64", "55", "45", "25"]

fig, ax = plt.subplots()

for file_suffix in suffixes:
    results_file_name = os.path.join("results", f"results_{file_suffix}.json")
    results = ph.ScanResults.from_json(filename=results_file_name)

    gf = np.array([r.discharge.greenwald_fraction for r in results.shots])
    lx = np.array([r.blob_params.lx_f for r in results.shots])
    ly = np.array([r.blob_params.ly_f for r in results.shots])
    ax.scatter(gf, lx / ly, label=file_suffix)

ax.legend()
ax.set_xlabel(r"$f_g$")
ax.set_ylabel(r"$\ell_x/\ell_y$")

file_name = os.path.join("result_plots", f"aspect_ratios.eps")
plt.savefig(file_name, bbox_inches="tight")
plt.close(fig)

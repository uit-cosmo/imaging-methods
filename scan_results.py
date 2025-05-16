import numpy as np

from phantom import ScanResults
import matplotlib.pyplot as plt
from cosmoplots import figure_multiple_rows_columns

results_contouring = ScanResults.from_json(
    "density_scan_results_contouring/results_contouring.json"
)
results_fit = ScanResults.from_json("density_scan_results_fit/results_fit.json")

# r = results.shots[0]

gf_f = np.array([r.plasma_discharge.greenwald_fraction for r in results_fit.shots])
taud_f = np.array([r.taud for r in results_fit.shots])
v_f = np.array([r.v for r in results_fit.shots])
lx_f = np.array([r.lx for r in results_fit.shots])

gf_c = np.array(
    [r.plasma_discharge.greenwald_fraction for r in results_contouring.shots]
)
taud_c = np.array([r.taud for r in results_contouring.shots])
v_c = np.array([r.v for r in results_contouring.shots])
lx_c = np.array([r.lx for r in results_contouring.shots])
lx_c = np.sqrt(lx_c)

fix, ax = figure_multiple_rows_columns(1, 1)
ax = ax[0]

# ax.scatter(gf, v/max(v), label=r"$v_\text{TDE}$", color="black")
# ax.scatter(gf, taud/max(taud), label=r"$\tau_d$", color="red")
# ax.scatter(gf, lx/max(lx), label=r"$\ell_x$", color="green")

ax.scatter(
    gf_c, v_c * taud_c / lx_c, label=r"$v_\text{C} \tau_d / \ell_c$", color="black"
)
ax.scatter(
    gf_f, v_f * taud_f / lx_f, label=r"$v_\text{TDE} \tau_d / \ell_f$", color="green"
)
ax.set_xlabel(r"$f_g$")
ax.set_ylabel(r"$v \tau_d / \ell$")

ax.legend()
ax.plot()
plt.savefig("ratios.png", bbox_inches="tight")
plt.show()

fix, ax = figure_multiple_rows_columns(1, 2)

ax[0].scatter(gf_c, v_c, label=r"$v_\text{C}$", color="black")
ax[0].scatter(gf_f, v_f, label=r"$v_\text{TDE}$", color="green")
ax[0].legend()
ax[0].set_xlabel(r"$f_g$")
ax[0].set_ylabel(r"$v$")
ax[0].set_ylim(0, 1000)

ax[1].scatter(gf_c, lx_c, label=r"$\ell_\text{C}$", color="black")
ax[1].scatter(gf_f, lx_f, label=r"$\ell_f$", color="green")
ax[1].legend()
ax[1].set_xlabel(r"$f_g$")
ax[1].set_ylabel(r"$\ell$")
ax[1].set_ylim(0, 0.01)

plt.savefig("velocities.png", bbox_inches="tight")
plt.show()

fix, ax = figure_multiple_rows_columns(1, 1)
ax = ax[0]

# ax.scatter(gf, v/max(v), label=r"$v_\text{TDE}$", color="black")
# ax.scatter(gf, taud/max(taud), label=r"$\tau_d$", color="red")
# ax.scatter(gf, lx/max(lx), label=r"$\ell_x$", color="green")

ax.scatter(gf_c, taud_c, label=r"$\tau_c$", color="black")
ax.scatter(gf_f, taud_f, label=r"$\tau_f$", color="green")
ax.set_xlabel(r"$f_g$")
ax.set_ylabel(r"$\tau_d$")

ax.legend()
ax.plot()
plt.savefig("taud.png", bbox_inches="tight")
plt.show()

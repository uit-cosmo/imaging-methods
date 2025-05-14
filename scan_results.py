import numpy as np

from phantom import ScanResults
import matplotlib.pyplot as plt

results = ScanResults.from_json("results.json")
results.print_summary()

# r = results.shots[0]

gf = np.array([r.plasma_discharge.greenwald_fraction for r in results.shots])
taud = np.array([r.taud for r in results.shots])
v = np.array([r.v for r in results.shots])
lx = np.array([r.lx for r in results.shots])

fig, ax = plt.subplots()

# ax.scatter(gf, v/max(v), label=r"$v_\text{TDE}$", color="black")
# ax.scatter(gf, taud/max(taud), label=r"$\tau_d$", color="red")
# ax.scatter(gf, lx/max(lx), label=r"$\ell_x$", color="green")

ax.scatter(gf, v * taud / lx, label=r"$v_\text{TDE} \tau_d / \ell_x$", color="black")
ax.set_xlabel(r"$f_g$")
ax.set_ylabel(r"$v_\text{TDE} \tau_d / \ell_x$")

ax.legend()
ax.plot()
plt.show()

import blobmodel as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import welch
import fppanalysis as fppa

import cosmoplots as cp

matplotlib_params = plt.rcParams
cp.set_rcparams_dynamo(matplotlib_params, 2)
plt.rcParams.update(matplotlib_params)


def get_analytical_acf(times, tau, shape):
    if shape == bm.BlobShapeEnum.exp:
        return tau * np.exp(-np.abs(times) / tau)
    if shape == bm.BlobShapeEnum.gaussian:
        return tau * np.exp(-1 / 2 * (times / tau) ** 2)
    return np.zeros(len(times))


shape = bm.BlobShapeEnum.exp

fig, ax = plt.subplots()

q = 1 / 2
s = 0.5

times = np.linspace(-3, 3, num=101)

acf_exp = q * get_analytical_acf(times, 1 - s, shape) + q * get_analytical_acf(
    times, 1 + s, shape
)
acf_exp_deg = get_analytical_acf(times, 1, shape)

acf_gauss = q * get_analytical_acf(
    times, 1 - s, bm.BlobShapeEnum.gaussian
) + q * get_analytical_acf(times, 1 + s, bm.BlobShapeEnum.gaussian)
acf_gauss_deg = get_analytical_acf(times, 1, bm.BlobShapeEnum.gaussian)

ax.plot(times, acf_exp_deg, color="blue")
ax.plot(times, acf_exp, color="blue", ls="--")
ax.plot(times, acf_gauss_deg, color="red")
ax.plot(times, acf_gauss, color="red", ls="--")

plt.show()

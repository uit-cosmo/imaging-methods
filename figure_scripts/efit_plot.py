import numpy as np

import matplotlib.pyplot as plt
from imaging_methods import *
import cosmoplots as cp

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 2)
# plt.rcParams["text.usetex"] = False
plt.rcParams.update(params)

shot = 1120814031
ds = manager.read_shot_data(shot, preprocessed=True)

t_start, t_end = (
    manager.get_discharge_by_shot(shot).t_start,
    manager.get_discharge_by_shot(shot).t_end,
)
r_min, r_max, z_fine = get_lcfs_min_and_max(ds, t_start, t_end)

fig, ax = plt.subplots()

im = ax.imshow(
    ds.frames.isel(time=149827).values,
    origin="lower",
    interpolation="spline16",
)
im.set_extent((ds.R[0, 0], ds.R[0, -1], ds.Z[0, 0], ds.Z[-1, 0]))
# ax.plot(r_min, z_fine, color="black")
# ax.plot(r_max, z_fine, color="black")
ax.fill_betweenx(
    z_fine,  # y-values (vertical axis)
    r_min,  # x1-values (lower boundary)
    r_max,  # x2-values (upper boundary)
    color="grey",  # Color of the shaded area
    alpha=0.5,  # Transparency (0 = transparent, 1 = opaque)
)
ax.set_ylim((ds.Z[0, 0], ds.Z[-1, 0]))
ax.set_xlim((ds.R[0, 0], ds.R[0, -1]))

plt.show()

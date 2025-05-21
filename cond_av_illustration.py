from phantom.cond_av import *
import matplotlib.pyplot as plt
import xarray as xr

shot = 1160616018
ds = xr.open_dataset("data18small.nc")

refx, refy = 6, 5

events, average = find_events(
    ds, refx, refy, threshold=2.5, window_size=120, check_max=1, single_counting=True
)

fig, axes = plt.subplots(3, 3)
e = events[1]
peak = e.isel(x=refx, y=refy).frames.max().item()
dt = 5e-7

for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        ax = axes[j + 1, i + 1]
        pixel = e.isel(x=refx + i, y=refy - j)
        ax.plot(e.time.values, pixel.frames.values)
        R, Z = pixel.R.item(), pixel.Z.item()
        ax.hlines(
            2.5,
            pixel.time.min().item(),
            pixel.time.max().item(),
            color="black",
            linewidth=0.5,
        )
        ax.vlines(0, -1, 2 * peak, linewidth=0.5, color="black")
        ax.set_title(r"$R = {:.2f}, Z = {:.2f}$".format(R, Z))
        ax.set_ylim(-1, peak * 1.1)
        ax.fill_between([-30 * dt, +30 * dt], -2, 2 * peak, color="lightgrey")

plt.savefig("event_illustration.png", bbox_inches="tight")
plt.show()

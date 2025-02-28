from phantom.show_data import *
from phantom.utils import *

shot = 1160616026

ds_phantom = xr.open_dataset("data/phantom_{}.nc".format(shot))

# Running mean
ds_phantom["frames"] = run_norm_ds(ds_phantom, 1000)["frames"]

t_start, t_end = get_t_start_end(ds_phantom)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.002
ds_phantom = ds_phantom.sel(time=slice(t_start, t_end))

# Roll mean in space
box_size = 5
ds_phantom = ds_phantom.rolling(
    x=box_size, y=box_size, center=True, min_periods=1
).mean()

ds_apd = xr.open_dataset("data/apd_{}.nc".format(shot))
ds_apd["frames"] = run_norm_ds(ds_apd, 1000)["frames"]
ds_apd = ds_apd.sel(time=slice(t_start, t_end))
interpolate_nans_3d(ds_apd)

has_limiter = "rlimt" in ds_phantom.coords.keys()
has_lcfs = "rlcfs" in ds_phantom


def animate_2d(i: int) -> Any:
    arr = ds_apd["frames"].isel({"time": i})
    vmin, vmax = -1, 3
    im_apd.set_data(arr)
    # im.set_extent((dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1]))
    im_apd.set_clim(vmin, vmax)
    time = ds_apd["time"][i].values
    tx_apd.set_text(f"t = {time:.7f}")
    plot_phantom(time)


def plot_phantom(time):
    index = np.absolute(time - ds_phantom["time"].values).argmin()
    arr = ds_phantom["frames"].isel({"time": index})
    im_phantom.set_data(arr)
    im_phantom.set_clim(-1, 3)
    time_phantom = ds_phantom["time"][index]
    tx_phantom.set_text(f"t = {time_phantom:.7f}")


fig, axes = plt.subplots(1, 2)
t_init = ds_apd["time"][0].values
tx_apd = axes[1].set_title(f"t = {t_init:.7f}")
im_apd = axes[1].imshow(
    ds_apd["frames"].isel({"time": 0}),
    origin="lower",
    interpolation="spline16",
)
index = np.absolute(t_init - ds_phantom["time"].values).argmin()
im_phantom = axes[0].imshow(
    ds_phantom["frames"].isel({"time": index}), origin="lower", interpolation="spline16"
)
phantom_time = ds_phantom["time"][index]
tx_phantom = axes[0].set_title(f"t = {phantom_time:.7f}")

im_apd.set_extent((ds_apd.R[0, -1], ds_apd.R[0, 0], ds_apd.Z[0, 0], ds_apd.Z[-1, 0]))
axes[0].plot(
    [ds_apd.R[0, -1], ds_apd.R[0, 0], ds_apd.R[0, 0], ds_apd.R[0, -1], ds_apd.R[0, -1]],
    [ds_apd.Z[0, 0], ds_apd.Z[0, 0], ds_apd.Z[-1, 0], ds_apd.Z[-1, 0], ds_apd.Z[0, 0]],
    color="black",
)
im_phantom.set_extent(
    (ds_phantom.R[0, -1], ds_phantom.R[0, 0], ds_phantom.Z[0, 0], ds_phantom.Z[-1, 0])
)

ani = animation.FuncAnimation(
    fig, animate_2d, frames=ds_apd["time"].values.size, interval=100
)

ani.save("output.gif", writer="ffmpeg", fps=60)
plt.show()

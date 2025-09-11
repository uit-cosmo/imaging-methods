from imaging_methods.utils import *
import velocity_estimation as ve
import cosmoplots as cp
import matplotlib.pyplot as plt

shot = 1120921007

ds = xr.open_dataset("data/phantom_1090813019.nc")
# ds = xr.open_dataset("~/phantom_1160616016.nc")
# ph_1160616016 goes 1.1 to 1.6
# ph_1120921007 goes 1.35 to 1.5

# Running mean
ds = run_norm_ds(ds, 1000)

# Roll mean in space
box_size = 5
ds = ds.rolling(x=box_size, y=box_size, center=True, min_periods=1).mean()

t_start, t_end = get_t_start_end(ds)
print("Data with times from {} to {}".format(t_start, t_end))

t_start = (t_start + t_end) / 2
t_end = t_start + 0.01
ds = ds.sel(time=slice(t_start, t_end))
dt = get_dt(ds)

px, py = 50, 32
ref_signal = ds.isel(x=px, y=py)["frames"].values


def get_tde_curve(p0, n_neighbours, direction, data):
    tdd = ve.TDEDelegator(ve.TDEMethod.CC, ve.CCOptions(interpolate=True), False)
    neighbours = [
        (p0[0] + i, p0[1]) if direction == "h" else (p0[0], p0[1] + i)
        for i in np.arange(-n_neighbours, n_neighbours + 1)
    ]
    taus_return, ctaus_return, _ = zip(
        *[
            tdd.estimate_time_delay(neighbour, p0, PhantomDataInterface(data))
            for neighbour in neighbours
        ]
    )
    return taus_return, ctaus_return


def plot_curves(p0, ax0, ax1):
    t, c = get_tde_curve(p0, 10, "h", ds)
    ax0.plot(t, c, label="{}".format(p0))
    t, c = get_tde_curve(p0, 10, "v", ds)
    ax1.plot(t, c, label="{}".format(p0))


fig, ax = cp.figure_multiple_rows_columns(1, 2)

plot_curves((50, 8), ax[0], ax[1])
plot_curves((50, 24), ax[0], ax[1])
plot_curves((50, 40), ax[0], ax[1])
plot_curves((50, 53), ax[0], ax[1])

ax[0].legend()
plt.show()

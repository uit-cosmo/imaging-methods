from phantom.show_data import *
from phantom.utils import *
import fppanalysis as fpp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import least_squares

shot = 1140613026

ds = xr.open_dataset("ds_short.nc")


def get_signal(i, j):
    return ds["frames"][i, j, :]


times = ds["time"].values


def plot_ccfs():
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 20))
    refx, refy = 7, 2
    ref_signal = get_signal(refy, refx)
    delta = 5e-5

    for row in range(4):
        ax[row].set_title("Col {}".format(5 + row))
        ax[row].vlines(0, -10, 10, ls="--")
        ax[row].set_ylim(-0.5, 1.1)
        for j in range(8):
            signal = get_signal(j, refx - 2 + row)
            time, res = fpp.corr_fun(signal, ref_signal, 5e-7)
            window = np.abs(time) < delta
            # Svals, s_av, s_var, t_av, peaks, wait = fpp.cond_av(S=get_signal(j, refx-2+row), T=times, smin=2, Sref=get_signal(refy, refx), delta=5e-5)
            ax[row].plot(
                time[window],
                res[window],
                label="Z={:.2f}".format(ds.Z.values[j, refx - 2 + row]),
            )

    ax[0].legend(loc=5)

    plt.savefig("ref72_ccf.eps", bbox_inches="tight")
    plt.show()


ds_short = ds.isel(x=slice(-4, None))
refx, refy = 2, 5

# Extract the reference time series
s_ref = ds_short.frames.isel(
    x=refx, y=refy
).values  # Select the time series at (refx, refy)
tau, _ = fpp.corr_fun(
    ds_short.frames.isel(x=0, y=0).values, s_ref, dt=5e-7
)  # Apply correlation function to each time series


def corr_wrapper(s):
    tau, res = fpp.corr_fun(
        s_ref, s, dt=5e-7
    )  # Apply correlation function to each time series
    return res


ds_corr = xr.apply_ufunc(
    corr_wrapper,
    ds_short,
    input_core_dims=[["time"]],  # Each function call operates on a single time series
    output_core_dims=[["tau"]],  # Output is also a time array
    vectorize=True,
)
ds_corr = ds_corr.assign_coords(tau=tau)
trajectory_times = tau[np.abs(tau) < 1e-5]
ds_corr = ds_corr.sel(tau=trajectory_times)

# Example 4x10 matrix (replace this with your actual data)
data = np.random.rand(4, 10)

# Generate x, y coordinate grids assuming uniform spacing
x_vals = np.linspace(-5, 5, 10)  # Adjust based on actual range
y_vals = np.linspace(-2, 2, 4)   # Adjust based on actual range
X, Y = np.meshgrid(x_vals, y_vals)

# Flatten matrices for fitting
x_data = X.ravel()
y_data = Y.ravel()
z_data = data.ravel()

# Define the function to fit
def model(params, x, y):
    lx, ly, rx, ry, t = params
    xt = (x - rx) * np.cos(t) + (y - ry) * np.sin(t)
    yt = (y - ry) * np.cos(t) - (x - rx) * np.sin(t)
    ret = np.exp(-((xt / lx) ** 2) - ((yt / ly) ** 2))
    return ret

# Define the residual function
def residuals(params, x, y, z):
    return np.sum(np.abs(model(params, x, y) - z))

# Initial guesses for lx, ly, and t
initial_guess = [1.0, 1.0, 90, 0, 0.0]

# Perform the least squares fit
data = ds_corr.sel(tau=0).frames.values
result = least_squares(residuals, initial_guess, args=(ds_corr.R.values, ds_corr.Z.values, data))

# Extract fitted parameters
lx_fit, ly_fit, rx_fit, ry_fit, t_fit = result.x

print(f"Fitted parameters: lx = {lx_fit:.4f}, ly = {ly_fit:.4f}, rx = {rx_fit:.4f}, ry = {ry_fit:.4f}, t = {t_fit:.4f}")

fig, ax = plt.subplots()

im = ax.imshow(ds_corr.sel(tau=0).frames, origin="lower", interpolation="spline16")
ax.scatter(ds_corr.isel(x=refx, y=refy).R, ds_corr.isel(x=refx, y=refy).Z, color="black")
im.set_extent(
    (ds_corr.R[0, 0]-0.05, ds_corr.R[0, -1], ds_corr.Z[0, 0], ds_corr.Z[-1, 0])
)

plt.show()

# Find the pixel with the largest value for each trajectory time
max_pixel_indices = []

for t in trajectory_times:
    data_at_t = ds_corr.sel(tau=t).to_array().squeeze()  # Convert to DataArray and remove extra dimensions

    # Find the index of the maximum value
    idx = np.unravel_index(np.argmax(data_at_t.values), data_at_t.shape)

    # Get the corresponding (x, y) coordinates
    max_pixel_indices.append((idx[1], idx[0]))

print(max_pixel_indices)

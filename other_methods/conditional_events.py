from phantom.data_preprocessing import load_data_and_preprocess
from phantom.utils import *
from phantom.cond_av import *
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import velocity_estimation as ve
from scipy import stats


shot = 1160616018
shot = 1140613026
ds = load_data_and_preprocess(shot, 0.2)
# ds.to_netcdf("data1426large.nc")
ds = xr.open_dataset("../data1426large.nc")

refx, refy = 6, 5

events, average = find_events_and_2dca(
    ds, refx, refy, threshold=2.5, check_max=1, single_counting=True
)

N = 1
length = 2 * N + 1
eindx = 0


def save_events(event_list):
    for e in event_list:
        e.to_netcdf("tmp/event_{}_{}.nc".format(shot, e["event_id"].item()))


save_events(events)


def plot_event(e, ax, indx):
    convolved_times_ref, convolved_data_ref = gaussian_convolve(
        e.frames.isel(x=refx, y=refy), e.time, s=3
    )
    max_time_reference, _ = find_maximum_interpolate(
        convolved_times_ref, convolved_data_ref
    )
    for i in range(length):
        for j in range(length):
            y, x = refy - j + N, refx + i - N
            axe = ax[j][i]
            data = e.frames.isel(x=x, y=y)
            _, convolved_data = gaussian_convolve(data, e.time, s=3)
            max_time, _ = find_maximum_interpolate(convolved_times_ref, convolved_data)
            axe.plot(e.time, data)
            axe.plot(convolved_times_ref, convolved_data)
            R = e.R.isel(x=x, y=y).item()
            Z = e.Z.isel(x=x, y=y).item()
            axe.set_title("R = {:.2f}, Z = {:.2f}".format(R, Z))
            if i == 1 and j == 1:
                taux, tauy = get_delays(e, refx, refy)
                text = "tx = {:.2f}, ty = {:.2f}".format(taux * 1e6, tauy * 1e6)
                print("Event {} {}".format(e["event_id"], text))
                axe.text(
                    e.time.mean(),
                    np.max(convolved_data) * 0.75,
                    s=text,
                )
            else:
                axe.text(
                    e.time.mean(),
                    np.max(convolved_data) * 0.75,
                    s="{:.2f}".format((max_time - max_time_reference) * 1e6),
                )
            plt.savefig("tmp/{}.png".format(indx), bbox_inches="tight")


deltax = (
    average.R.isel(x=refx + 1, y=refy).item() - average.R.isel(x=refx, y=refy).item()
)
deltay = (
    average.Z.isel(x=refx, y=refy + 1).item() - average.Z.isel(x=refx, y=refy).item()
)

rmin, rmax, zmin, zmax = (
    average.R[0, 0] - 0.05,
    average.R[0, -1] + 0.05,
    average.Z[0, 0] - 0.05,
    average.Z[-1, 0] + 0.05,
)
R, Z = average.R.isel(x=refx, y=refy).item(), average.Z.isel(x=refx, y=refy).item()


def plot_fit_ellipse():
    fig, ax = plt.subplots(4, 4)
    for i in range(16):
        axe = ax[int(i / 4)][i % 4]
        e = events[i]
        lx, ly, theta = fit_ellipse(
            e.sel(time=0), R, Z, size_penalty_factor=5, aspect_ratio_penalty_factor=1
        )
        im = axe.imshow(e.sel(time=0).frames, origin="lower", interpolation="spline16")
        alphas = np.linspace(0, 2 * np.pi, 200)
        elipsx, elipsy = zip(
            *[ellipse_parameters((lx, ly, theta), R, Z, a) for a in alphas]
        )
        axe.plot(elipsx, elipsy)
        im.set_extent((rmin, rmax, zmin, zmax))

    plt.savefig("event_fits_size_aspect_penalty.png", bbox_inches="tight")


plot_fit_ellipse()


plot = False
for e in events:
    taux, tauy = get_delays(e, refx, refy)
    amplitude = get_maximum_amplitude(e, refx, refy)
    e["taux"] = taux
    e["tauy"] = tauy
    e["amplitude"] = amplitude
    if amplitude < 2:
        print("LOL")
    v, w = ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)
    e["u"] = np.sqrt(v**2 + w**2) / 100
    e["v"] = v / 100
    e["w"] = w / 100
    lx, ly, theta = fit_ellipse(
        e.sel(time=0), R, Z, size_penalty_factor=5, aspect_ratio_penalty_factor=1
    )
    e["lx"] = lx
    e["ly"] = ly
    e["theta"] = theta
    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(e.sel(time=0).frames, origin="lower", interpolation="spline16")
        alphas = np.linspace(0, 2 * np.pi, 200)
        elipsx, elipsy = zip(
            *[ellipse_parameters((lx, ly, theta), R, Z, a) for a in alphas]
        )
        ax.plot(elipsx, elipsy)
        im.set_extent((rmin, rmax, zmin, zmax))
        plt.savefig("tmp/fit_{}.png".format(e["event_id"].item()), bbox_inches="tight")

amplitudes = np.array([e["amplitude"] for e in events])
vs = np.array([e["v"] for e in events])
ws = np.array([e["w"] for e in events])
us = np.array([e["u"] for e in events])
lxs = np.array([e["lx"] for e in events])
lys = np.array([e["ly"] for e in events])
ells = np.sqrt(lxs * lys)
thetas = np.array([e["theta"] for e in events])

print(
    "Events {}, Mean v {:.2f}, mean w {:.2f}".format(
        len(amplitudes), vs.mean(), ws.mean()
    )
)

fig, ax = plt.subplots()

sigma_v, sigma_u, sigma_ell = (
    stats.pearsonr(amplitudes, vs),
    stats.pearsonr(amplitudes, us),
    stats.pearsonr(amplitudes, ells),
)
print(
    r"$\sigma_v = {:.2f}, \sigma_u = {:.2f}$, \sigma_\ell = {:.2f}".format(
        sigma_v.statistic, sigma_u.statistic, sigma_ell.statistic
    )
)
ax.scatter(ells, amplitudes)
# ax.set_xlabel(r"$u_{\text{TDE}}$")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$a$")
plt.savefig("tmp/ell_corr.png", bbox_inches="tight")
plt.show()

print("LOL")

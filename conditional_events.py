import numpy as np
from synthetic_data import *
from phantom.show_data import show_movie
from phantom.utils import *
from phantom.cond_av import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import warnings
import velocity_estimation as ve


shot = 1160616018
#ds = get_sample_data(shot, 0.1)
#ds.to_netcdf("data.nc")
ds = xr.open_dataset("data.nc")

refx, refy = 6, 5

events, average = find_events(
    ds, refx, refy, threshold=2.5, check_max=1, single_counting=True
)

N = 1
length = 2 * N + 1
eindx = 0


def save_events(event_list):
    for e in event_list:
        e.to_netcdf("tmp/event_{}_{}.nc".format(shot, e["event_id"].item()))


save_events(events)


def gaussian_convolve(x, times, s=1.0, kernel_size=None):
    # If kernel_size not specified, use 6*sigma to capture most of the Gaussian
    if kernel_size is None:
        kernel_size = int(6 * s)
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

    center = kernel_size // 2
    kernel = np.exp(-((np.arange(-center, center + 1) / s) ** 2))
    kernel = kernel / kernel.sum()

    return times[center:-center], np.convolve(x, kernel, mode="valid")


def find_maximum_interpolate(x, y):
    from scipy.interpolate import InterpolatedUnivariateSpline

    # Taking the derivative and finding the roots only work if the spline degree is at least 4.
    spline = InterpolatedUnivariateSpline(x, y, k=4)
    possible_maxima = spline.derivative().roots()
    possible_maxima = np.append(
        possible_maxima, (x[0], x[-1])
    )  # also check the endpoints of the interval
    values = spline(possible_maxima)

    max_index = np.argmax(values)
    max_time = possible_maxima[max_index]
    if max_time == x[0] or max_time == x[-1]:
        warnings.warn(
            "Maximization on interpolation yielded a maximum in the boundary!"
        )

    return max_time, spline(max_time)


def get_maximum_time(e, x, y):
    convolved_times, convolved_data = gaussian_convolve(
        e.frames.isel(x=x, y=y), e.time, s=3
    )
    tau, _ = find_maximum_interpolate(convolved_times, convolved_data)
    return tau


def get_maximum_amplitude(e, x, y):
    convolved_times, convolved_data = gaussian_convolve(
        e.frames.isel(x=x, y=y), e.time, s=3
    )
    _, amp = find_maximum_interpolate(convolved_times, convolved_data)
    return amp


def get_delays(e, refx, refy):
    ref_time = get_maximum_time(e, refx, refy)
    taux_right = get_maximum_time(e, refx+1, refy) - ref_time
    taux_left = get_maximum_time(e, refx-1, refy) - ref_time
    tauy_up = get_maximum_time(e, refx, refy+1) - ref_time
    tauy_down = get_maximum_time(e, refx, refy-1) - ref_time
    return (taux_right - taux_left) / 2, (tauy_up - tauy_down) / 2


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

deltax = average.R.isel(x=refx+1, y=refy).item() - average.R.isel(x=refx, y=refy).item()
deltay = average.Z.isel(x=refx, y=refy+1).item() - average.Z.isel(x=refx, y=refy).item()


for e in events:
    taux, tauy = get_delays(e, refx, refy)
    amplitude = get_maximum_amplitude(e, refx, refy)
    e["taux"] = taux
    e["tauy"] = tauy
    e["amplitude"] = amplitude
    if amplitude < 2:
        print("LOL")
    v, w = ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)
    e["u"] = np.sqrt(v**2+w**2) / 100
    e["v"] = v / 100
    e["w"] = w / 100

amplitudes = np.array([e["amplitude"] for e in events])
vs = np.array([e["v"] for e in events])
ws = np.array([e["w"] for e in events])
us = np.array([e["u"] for e in events])

print("Events {}, Mean v {:.2f}, mean w {:.2f}".format(len(amplitudes), vs.mean(), ws.mean()))

fig, ax = plt.subplots()

ax.scatter(vs, amplitudes)
ax.set_xlabel(r"$v_{\text{TDE}}$")
ax.set_ylabel(r"$a$")
plt.show()

print("LOL")

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


# ds = get_sample_data(1160616025, 0.01)
# ds.to_netcdf("data.nc")
shot = 1160616025
ds = xr.open_dataset("data.nc")

refx, refy = 6, 5

events, average = find_events(
    ds, refx, refy, threshold=2, check_max=1, single_counting=True
)

N = 1
length = 2 * N + 1
eindx = 0

# def get_tde_event(e):
import numpy as np


def save_events(event_list):
    for e in event_list:
        e.to_netcdf("tmp/event_{}_{}.nc".format(shot, e["event_id"].item()))


# save_events(events)


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

    return max_time


def plot_event(e, ax, indx):
    convolved_times_ref, convolved_data_ref = gaussian_convolve(
        e.frames.isel(x=refx, y=refy), e.time, s=3
    )
    max_time_reference = find_maximum_interpolate(
        convolved_times_ref, convolved_data_ref
    )
    for i in range(length):
        for j in range(length):
            y, x = refy - j + N, refx + i - N
            axe = ax[j][i]
            data = e.frames.isel(x=x, y=y)
            _, convolved_data = gaussian_convolve(data, e.time, s=3)
            max_time = find_maximum_interpolate(convolved_times_ref, convolved_data)
            axe.plot(e.time, data)
            axe.plot(convolved_times_ref, convolved_data)
            R = e.R.isel(x=x, y=y).item()
            Z = e.Z.isel(x=x, y=y).item()
            axe.set_title("R = {:.2f}, Z = {:.2f}".format(R, Z))
            axe.text(
                e.time.mean(),
                np.max(convolved_data) * 0.75,
                s="{:.2f}".format((max_time - max_time_reference) * 1e6),
            )
            plt.savefig("tmp/{}.png".format(indx), bbox_inches="tight")


for e in events:
    fig, ax = plt.subplots(length, length)
    plot_event(e, ax, eindx)
    eindx += 1

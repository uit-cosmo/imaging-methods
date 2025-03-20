import numpy as np
from synthetic_data import *
from phantom.show_data import show_movie
from phantom.utils import *
from blobmodel import BlobShapeEnum
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd


def get_blob(vx, vy, posx, posy, lx, ly, t_init, theta, bs=BlobShapeImpl()):
    return Blob(
        1,
        bs,
        amplitude=1,
        width_prop=lx,
        width_perp=ly,
        v_x=vx,
        v_y=vy,
        pos_x=posx,
        pos_y=posy,
        t_init=t_init,
        t_drain=1e100,
        theta=theta,
        blob_alignment=True if theta == 0 else False,
    )


num_blobs = 100
T = 200
Lx = 3
Ly = 3
lx = 0.5
ly = 1.5
nx = 20
ny = 20
vx = 1
vy = 1
theta = -np.pi/4
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)

blobs = [get_blob(vx=vx, vy=vy, posx=np.random.uniform(0, Lx), posy=Ly, lx=lx, ly=ly, t_init=np.random.uniform(0, T), bs=bs, theta=theta) for _ in range(num_blobs)]

rp = RunParameters(T=T, lx=Lx, ly=Ly, nx=nx, ny=ny)
bf = DeterministicBlobFactory(blobs)

ds = make_2d_realization(rp, bf)

refx, refy = 10, 10


def find_events(ds, refx, refy, threshold=3, window_size=10):
    """
    Find events where reference pixel exceeds threshold and extract windows around peaks.

    Parameters:
    ds (xarray.Dataset): Input dataset with time, x, y coordinates
    refx (int): X index of reference pixel
    refy (int): Y index of reference pixel
    threshold (float): Threshold value for event detection
    window_size (int): Size of window to extract around peaks

    Returns:
    list: List of xarray.Dataset objects containing extracted windows
    """
    # Assuming the data variable is named 'data' - adjust if different
    ref_ts = ds.frames.isel(x=refx, y=refy)

    # Find indices where signal exceeds threshold
    above_threshold = ref_ts > threshold
    indices = np.where(above_threshold)[0]

    # Split into contiguous events
    events = []
    if len(indices) > 0:
        diffs = np.diff(indices)
        split_points = np.where(diffs > 1)[0] + 1
        events = np.split(indices, split_points)

    windows = []
    half_window = window_size // 2

    for event in events:
        if len(event) == 0:
            continue

        # Find peak within the event
        event_ts = ref_ts.isel(time=event)
        max_idx_in_event = event_ts.argmax().item()
        peak_time_idx = event[max_idx_in_event]

        global_max = True
        if global_max:
            peak = event_ts[max_idx_in_event]
            if not peak == ds.isel(time=peak_time_idx).max():
                continue

        # Calculate window bounds
        start = max(0, peak_time_idx - half_window)
        end = min(len(ds.time), peak_time_idx + half_window + 1)  # +1 for inclusive end

        # Skip incomplete windows if needed (optional)
        if (end - start) < window_size:
            continue

        # Extract window for all pixels
        window = ds.isel(time=slice(start, end))
        windows.append(window)

    return windows


def compute_average_event(windows):
    """
    Compute average event across all windows by aligning peak times.

    Parameters:
    windows (list of xarray.Dataset): List of event windows from find_events_and_extract_windows

    Returns:
    xarray.Dataset: Dataset containing average event across all input events
    """
    processed = []
    for win in windows:
        # Create relative time coordinates centered on peak
        time_length = win.sizes['time']
        half_window = (time_length - 1) // 2
        relative_time = np.arange(time_length) - half_window

        # Assign new time coordinates
        win = win.assign_coords(time=relative_time)
        processed.append(win)

    # Combine all events along new dimension and compute mean
    return xr.concat(processed, dim='event').mean(dim='event')

events = find_events(ds, refx, refy, threshold=0.2)
average = compute_average_event(events)

fig, ax = plt.subplots()

im = ax.imshow(average.sel(time=0).frames, origin="lower", interpolation="spline16")
plt.show()
print("LOL")
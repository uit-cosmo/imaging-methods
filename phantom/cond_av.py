import numpy as np
import xarray as xr


def find_events(ds, refx, refy, threshold=3, window_size=30, check_max=0):
    """
    Find events where reference pixel exceeds threshold and extract windows around peaks.

    Parameters:
    ds (xarray.Dataset): Input dataset with time, x, y coordinates
    refx (int): X index of reference pixel
    refy (int): Y index of reference pixel
    threshold (float): Threshold value for event detection
    window_size (int): Size of window to extract around peaks

    Returns:
    events: List of xarray.Dataset objects containing extracted windows
    average: xarray.Dataset containing average event across all input events
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

    print("Found {} events".format(len(events)))
    windows = []
    half_window = window_size // 2
    discarded_events_zero_len = 0
    discarded_events_not_max = 0
    discarded_events_truncated = 0

    for event in events:
        if len(event) == 0:
            discarded_events_zero_len += 1
            continue

        # Find peak within the event
        event_ts = ref_ts.isel(time=event)
        max_idx_in_event = event_ts.argmax().item()
        peak_time_idx = event[max_idx_in_event]

        if check_max != 0:
            ref_peak = ds.frames.isel(time=peak_time_idx, x=refx, y=refy).item()
            fromx = max(refx - check_max, 0)
            tox = min(refx + check_max, ds.sizes["x"] - 1)
            fromy = max(refy - check_max, 0)
            toy = min(refy + check_max, ds.sizes["y"] - 1)
            global_peak = (
                ds.frames.isel(
                    time=peak_time_idx, x=slice(fromx, tox), y=slice(fromy, toy)
                )
                .max()
                .item()
            )
            if not np.isclose(ref_peak, global_peak, atol=1e-6):
                discarded_events_not_max += 1
                continue

        # Calculate window bounds
        start = max(0, peak_time_idx - half_window)
        end = min(len(ds.time), peak_time_idx + half_window + 1)  # +1 for inclusive end

        # Skip incomplete windows if needed (optional)
        if (end - start) < window_size:
            discarded_events_truncated += 1
            continue

        # Extract window for all pixels
        window = ds.isel(time=slice(start, end))
        windows.append(window)

    print(
        "Discarded {} events. Not max {}, zero len {}, truncation {}".format(
            discarded_events_not_max
            + discarded_events_zero_len
            + discarded_events_truncated,
            discarded_events_not_max,
            discarded_events_zero_len,
            discarded_events_truncated,
        )
    )

    processed = []
    for win in windows:
        # Create relative time coordinates centered on peak
        time_length = win.sizes["time"]
        half_window = (time_length - 1) // 2
        relative_time = np.arange(time_length) - half_window

        # Assign new time coordinates
        win = win.assign_coords(time=relative_time)
        processed.append(win)

    # Combine all events along new dimension and compute mean
    average = xr.concat(processed, dim="event").mean(dim="event")

    return windows, average

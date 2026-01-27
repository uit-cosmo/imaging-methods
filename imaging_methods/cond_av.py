import numpy as np
import xarray as xr
import scipy.signal as ssi
from .method_parameters import TwoDcaParams


def find_events_and_2dca(
    ds,
    params: TwoDcaParams,
    verbose=True,
):
    """
    2DCA method as described in the article.
    Find events where reference pixel exceeds threshold and extract windows around peaks.

    Parameters:
    ds (xarray.Dataset): Input dataset with time, x, y coordinates. cmod_functions format is expected.
    params (TwoDcaParams): Dataclass with method settings, documentation of each setting is available on TwoDcaParams
    verbose (bool): If True, print some method information.

    Returns:
    tuple: A tuple containing two elements:
        - events (list of xarray.Dataset): A list of datasets, each representing a time window around a detected event peak.
            Each dataset contains:
                - Data variables:
                    - frames: Array of shape (time, x, y) containing the data for the event window.
                - Attributes:
                    - refx (int): X index of the reference pixel.
                    - refy (int): Y index of the reference pixel.
                    - event_id (int): Unique identifier for the event.
                    - abs_time (float): Absolute time of the event's peak.
                - Coordinates:
                    - time: Relative time coordinates centered on the peak (in units of the input dataset's time step),
                        ranging from -half_window * dt to +half_window * dt.
                    - x, y: Spatial coordinates inherited from the input dataset.
        - average (xarray.Dataset): A dataset containing the conditional average and related metrics across all events.
            Contains:
                - Data variables:
                    - cond_av: Array of shape (time, x, y) representing the mean of all event windows.
                    - cond_repr: Array of shape (time, x, y) representing the conditional representativeness,
                        calculated as cond_av^2 / mean2, where mean2 is the mean of squared event windows.
                    - cross_corr: Array of shape (time, x, y) with the estimated two-dimensional cross-correlation.
                - Coordinates:
                    - time: Relative time coordinates centered on the peak, matching the events' time coordinates.
                    - x, y: Spatial coordinates inherited from the input dataset.
                - Attributes:
                    - refx (int): X index of the reference pixel.
                    - refy (int): Y index of the reference pixel.
                    - number_events (int): The total number of valid events included in the average.
            If no valid events are found, returns an empty xarray.Dataset.
    """
    window_size = params.window
    if window_size % 2 == 0:
        window_size += 1
    half_window = (window_size - 1) // 2
    dt = float(ds["time"][1].values - ds["time"][0].values)
    rel_times_idx = np.arange(window_size) - half_window
    rel_times = rel_times_idx * dt

    ref_ts = ds.frames.isel(x=params.refx, y=params.refy)

    above_threshold = ref_ts > params.threshold
    indices = np.where(above_threshold)[0]

    events = []
    if len(indices) > 0:
        diffs = np.diff(indices)
        split_points = np.where(diffs > 1)[0] + 1
        events = np.split(indices, split_points)

    if verbose:
        print("Found {} events".format(len(events)))
    candidate_events = []
    discarded_events_zero_len = 0
    discarded_events_not_max = 0
    discarded_events_truncated = 0
    discarded_events_single_count = 0

    for event in events:
        if len(event) == 0:
            discarded_events_zero_len += 1
            continue

        event_ts = ref_ts.isel(time=event)
        max_idx_in_event = event_ts.argmax().item()
        peak_time_idx = event[max_idx_in_event]

        if params.check_max != 0:
            ref_peak = ds.frames.isel(
                time=peak_time_idx, x=params.refx, y=params.refy
            ).item()
            fromx = max(params.refx - params.check_max, 0)
            tox = min(params.refx + params.check_max, ds.sizes["x"])
            fromy = max(params.refy - params.check_max, 0)
            toy = min(params.refy + params.check_max, ds.sizes["y"])
            global_peak = (
                ds.frames.isel(
                    time=peak_time_idx, x=slice(fromx, tox + 1), y=slice(fromy, toy + 1)
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

        # Store candidate event
        candidate_events.append(
            {
                "peak_time": peak_time_idx,
                "peak_value": ref_ts.isel(time=peak_time_idx).item(),
                "start": start,
                "end": end,
            }
        )

    if params.single_counting:
        # Sort candidates by peak value descending
        candidate_events.sort(key=lambda x: -x["peak_value"])
        selected_events = []

        for candidate in candidate_events:
            conflict = False
            # Check against already selected events
            for selected in selected_events:
                if abs(candidate["peak_time"] - selected["peak_time"]) < window_size:
                    conflict = True
                    discarded_events_single_count += 1
                    break
            if not conflict:
                selected_events.append(candidate)

        candidate_events = selected_events

    windows = []
    for candidate in candidate_events:
        window = ds.isel(time=slice(candidate["start"], candidate["end"]))
        windows.append(window)

    if verbose:
        print(
            "Discarded {} events. Not max {}, zero len {}, truncation {}, single count {}".format(
                discarded_events_not_max
                + discarded_events_zero_len
                + discarded_events_truncated
                + discarded_events_single_count,
                discarded_events_not_max,
                discarded_events_zero_len,
                discarded_events_truncated,
                discarded_events_single_count,
            )
        )

    # === Spatiotemporal cross-correlation on full dataset ===
    nx = ds.sizes["x"]
    ny = ds.sizes["y"]
    n_lags = window_size

    # Preallocate
    corr_array = np.zeros((ny, nx, n_lags))

    # Indices for slicing
    start_idx = ds.sizes["time"] - half_window - 1
    end_idx = ds.sizes["time"] + half_window

    ref_ts_norm = (ref_ts.values - ref_ts.values.mean()) / ref_ts.values.std()
    for i in range(nx):
        for j in range(ny):
            pixel = ds.frames.isel(x=i, y=j).values
            pixel = (pixel - pixel.mean()) / pixel.std()
            cov_sums_full = ssi.correlate(pixel, ref_ts_norm, mode="full")
            corr_array[j, i, :] = cov_sums_full[start_idx:end_idx] / len(ref_ts)

    cross_corr = xr.DataArray(
        corr_array,
        dims=["y", "x", "time"],
        coords={
            "R": (["y", "x"], ds.R.values),
            "Z": (["y", "x"], ds.Z.values),
            "time": (["time"], rel_times),
        },
        name="cross_corr",
    )

    # Processed events are the same as events in windows but with a time base relative to their maximum,
    # to make averaging possible
    processed = []
    event_id = 0
    for win in windows:
        abs_time = win.time[half_window].item()

        # Assign new time coordinates
        win = win.assign_coords(time=rel_times)
        win["event_id"] = event_id
        win["abs_time"] = abs_time
        event_id += 1
        processed.append(win)

    # Combine all events along new dimension and compute mean
    if len(processed) != 0:
        conditional_average = xr.concat(processed, dim="event").mean(dim="event").frames
        mean2 = (
            xr.apply_ufunc(lambda x: x**2, xr.concat(processed, dim="event"))
            .mean(dim="event")
            .frames
        )
        cond_av_ds = xr.Dataset(
            {
                "cond_av": conditional_average,
                "cond_repr": conditional_average**2 / mean2,
                "cross_corr": cross_corr,
            }
        )
        cond_av_ds["refx"] = params.refx
        cond_av_ds["refy"] = params.refy
        cond_av_ds["number_events"] = len(processed)
        return processed, cond_av_ds

    return processed, xr.Dataset()

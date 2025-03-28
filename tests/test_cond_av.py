import phantom as ph
import numpy as np
import xarray as xr


def get_single_pixel_ds(signal):
    times = np.arange(len(signal))

    r, z = 0, 0
    grid_r, grid_z = np.meshgrid(r, z)
    return xr.Dataset(
        {"frames": (["y", "x", "time"], signal[np.newaxis, np.newaxis, :])},
        coords={
            "R": (["y", "x"], grid_r),
            "Z": (["y", "x"], grid_z),
            "time": (["time"], times),
        },
    )


def get_multi_pixel_ds(signal1, signal2, signal3, signal4):
    times = np.arange(len(signal1))

    r, z = np.array([0, 1]), np.array([0, 1])
    grid_r, grid_z = np.meshgrid(r, z)
    return xr.Dataset(
        {
            "frames": (
                ["y", "x", "time"],
                np.array([[signal1, signal2], [signal3, signal4]]),
            )
        },
        coords={
            "R": (["y", "x"], grid_r),
            "Z": (["y", "x"], grid_z),
            "time": (["time"], times),
        },
    )


def test_threshold():
    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 1.2, 1]))
    events, average = ph.find_events(ds, 0, 0, threshold=0.5, window_size=3)
    assert len(events) == 2


def test_single_count():
    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 1, 1.2, 0, 0]))
    events, _ = ph.find_events(ds, 0, 0, threshold=0.5, window_size=4)
    assert len(events) == 2

    events, _ = ph.find_events(
        ds, 0, 0, threshold=0.5, window_size=5, single_counting=True
    )
    assert len(events) == 1

    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 1.2, 1, 0, 0, 1.2, 1, 0, 0]))
    events, _ = ph.find_events(
        ds, 0, 0, threshold=0.5, window_size=5, single_counting=True
    )
    assert len(events) == 2


def test_truncation():
    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 0, 0, 0, 0, 1, 1.2]))
    events, _ = ph.find_events(
        ds, 0, 0, threshold=0.5, window_size=5, single_counting=True
    )
    assert len(events) == 1


def test_check_max():
    signal1 = np.array([0, 0, 1, 3, 1, 0, 0])
    signal2 = np.array([0, 0, 1, 2, 1, 0, 0])
    signal3 = np.array([0, 0, 1, 2, 1, 0, 0])
    signal4 = np.array([0, 0, 1, 2, 1, 0, 0])

    ds = get_multi_pixel_ds(signal1, signal2, signal3, signal4)
    events, _ = ph.find_events(ds, 0, 0, threshold=0.5, window_size=5, check_max=1)
    assert len(events) == 1

    events, _ = ph.find_events(ds, 0, 0, threshold=0.5, window_size=5, check_max=0)
    assert len(events) == 1

    events, _ = ph.find_events(ds, 0, 1, threshold=0.5, window_size=5, check_max=1)
    assert len(events) == 0

    events, _ = ph.find_events(ds, 0, 1, threshold=0.5, window_size=5, check_max=0)
    assert len(events) == 1

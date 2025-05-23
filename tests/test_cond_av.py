import phantom as ph
import numpy as np
import xarray as xr
import pytest


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


@pytest.fixture
def synthetic_dataset():
    """Create a test dataset with known characteristics"""
    time = np.arange(100)
    x = y = np.arange(3)
    grid_x, grid_y = np.meshgrid(x, y)

    data = np.zeros((3, 3, 100))

    # Create reference pixel (1,1) with known events
    data[1, 1, 10:15] = 4  # Event 1
    data[1, 1, 30:35] = 5  # Event 2 (stronger)
    data[1, 1, 50:55] = 2  # Below threshold (2 < 3)
    data[1, 1, 70:75] = 4  # Event 3

    # Add a competing global maximum at 30-35
    data[2, 2, 30:35] = 6

    # Add overlapping events for single-counting test
    data[1, 1, 80:82] = 4.6  # Event 4
    data[1, 1, 84:88] = 5.1  # Overlapping Event 5

    return xr.Dataset(
        {"frames": (("y", "x", "time"), data)},
        coords={
            "R": (["y", "x"], grid_x),
            "Z": (["y", "x"], grid_y),
            "time": (["time"], time),
        },
    )


def test_threshold():
    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 1.2, 1]))
    events, average, _ = ph.find_events_and_2dca(ds, 0, 0, threshold=0.5, window_size=3)
    assert len(events) == 2


def test_single_count():
    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 1, 1.2, 0, 0]))
    events, _, _ = ph.find_events_and_2dca(ds, 0, 0, threshold=0.5, window_size=4)
    assert len(events) == 2

    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 0, threshold=0.5, window_size=5, single_counting=True
    )
    assert len(events) == 1

    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 1.2, 1, 0, 0, 1.2, 1, 0, 0]))
    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 0, threshold=0.5, window_size=5, single_counting=True
    )
    assert len(events) == 2


def test_truncation():
    ds = get_single_pixel_ds(np.array([0, 0, 1, 1.2, 0, 0, 0, 0, 0, 0, 1, 1.2]))
    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 0, threshold=0.5, window_size=5, single_counting=True
    )
    assert len(events) == 1


def test_conditional_reproducibility():
    signal = np.repeat(np.array([0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0]), 10)
    ds = get_single_pixel_ds(signal)
    events, _, cr = ph.find_events_and_2dca(
        ds, 0, 0, threshold=2.5, window_size=5, single_counting=True
    )
    assert np.all(cr.frames.isel(x=0, y=0).values == 1)


def test_conditional_reproducibility_change():
    signal = np.repeat(
        np.array([0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 3, 2, 0, 0, 0, 0]),
        10,
    )
    ds = get_single_pixel_ds(signal)
    events, _, cr = ph.find_events_and_2dca(
        ds, 0, 0, threshold=2.5, window_size=5, single_counting=True
    )
    assert np.all(cr.frames.isel(x=0, y=0).values < 1)


def test_check_max():
    signal1 = np.array([0, 0, 1, 3, 1, 0, 0])
    signal2 = np.array([0, 0, 1, 2, 1, 0, 0])
    signal3 = np.array([0, 0, 1, 2, 1, 0, 0])
    signal4 = np.array([0, 0, 1, 2, 1, 0, 0])

    ds = get_multi_pixel_ds(signal1, signal2, signal3, signal4)
    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 0, threshold=0.5, window_size=5, check_max=1
    )
    assert len(events) == 1

    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 0, threshold=0.5, window_size=5, check_max=0
    )
    assert len(events) == 1

    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 1, threshold=0.5, window_size=5, check_max=1
    )
    assert len(events) == 0

    events, _, _ = ph.find_events_and_2dca(
        ds, 0, 1, threshold=0.5, window_size=5, check_max=0
    )
    assert len(events) == 1


def test_basic_event_detection(synthetic_dataset):
    events, _, _ = ph.find_events_and_2dca(
        synthetic_dataset, 1, 1, threshold=3, window_size=5
    )
    assert len(events) == 5


def test_global_max_filter(synthetic_dataset):
    events, _, _ = ph.find_events_and_2dca(
        synthetic_dataset, 1, 1, threshold=3, window_size=5, check_max=2
    )
    # Event at 30-35 should be filtered out (global max at 2,2)
    assert len(events) == 4
    peak_times = [e.time.values[e.sizes["time"] // 2] for e in events]
    assert 32 not in peak_times  # Original peak time of filtered event


def test_single_counting(synthetic_dataset):
    events, _, _ = ph.find_events_and_2dca(
        synthetic_dataset, 1, 1, threshold=3, window_size=5, single_counting=True
    )
    assert len(events) == 4  # Original 3 valid events minus 1 overlap

    # Check spacing between events
    peaks = [e["abs_time"] for e in events]
    diffs = np.diff(sorted(peaks))
    assert all(d >= 5 for d in diffs)  # Window size spacing


def test_window_integrity(synthetic_dataset):
    events, _, _ = ph.find_events_and_2dca(
        synthetic_dataset, 1, 1, threshold=3, window_size=5
    )
    for event in events:
        assert event.sizes["time"] == 5
        # Check reference pixel is at center of window
        center_idx = event.sizes["time"] // 2
        assert event["frames"][1, 1, center_idx] >= 3


def test_edge_cases():
    # Test dataset with events near boundaries
    data = np.zeros(10)
    data[0:2] = 4  # Too close to start
    data[8:10] = 4  # Too close to end

    ds = get_single_pixel_ds(data)

    events, _, _ = ph.find_events_and_2dca(ds, 0, 0, window_size=5)
    assert len(events) == 0  # Should filter out both edge cases


def test_empty_result():
    ds = get_single_pixel_ds(np.zeros(100))
    events, _, _ = ph.find_events_and_2dca(ds, 0, 0, threshold=1)
    assert len(events) == 0


def test_single_counting_priority(synthetic_dataset):
    events, _, _ = ph.find_events_and_2dca(
        synthetic_dataset, 1, 1, threshold=3, window_size=5, single_counting=True
    )
    peaks = [e["abs_time"] for e in events]
    assert 84 in peaks
    assert 81 not in peaks  # Weaker overlapping event

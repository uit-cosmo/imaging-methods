"""imaging_methods.decorrelation on the synthetic record in tests/data/, against what the
code it was moved from gave there.

Moved from fusion_scripts `decorrelation/tests/test_pipeline.py` at 12b8543. The
expected values were made by the pre-refactor code (fusion_scripts
`decorrelation/tests/make_pipeline_fixture.py`, at 7b761fd), at settings that move one
knob in every group (pipeline_fixture.py). The comparison is the regression's: the same
NaN pattern and values within 1e-12 relative; in practice they are bit-equal.
"""

import dataclasses
import subprocess
import sys
import typing
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pipeline_fixture as fixture
from imaging_methods import (
    ContouringParams,
    MethodParameters,
    TwoDcaParams,
    decorrelation,
    find_events_and_2dca,
    get_default_apd_method_params,
    neighbour_level,
)

# The record, its settings, mask, averages, bank and products (fields_out, blobs_out)
# are conftest.py's.

# The hand-made mask of run day 1160616, as experimental_database holds it: one row per
# y, the top row (y = 9) first.
HAND_MADE_1160616 = (
    (True, True, False, True, False, False, True, True, True),
    (False, False, False, False, False, False, False, True, True),
    (True, False, False, False, True, False, False, False, True),
    (False, False, False, True, False, False, False, False, False),
    (True, False, False, False, False, False, False, False, True),
    (False, False, False, False, False, False, False, False, True),
    (False, False, False, False, False, False, False, True, False),
    (False, True, False, True, False, False, False, False, True),
    (True, True, False, False, False, False, False, False, False),
    (False, False, True, False, False, False, False, False, False),
)


def hand_made_1160616():
    """The 1160616 mask as experimental_database's ``hand_made_dead_pixel_mask`` gives
    it: bool (y, x), y counted from the bottom row."""
    return xr.DataArray(
        [list(row) for row in HAND_MADE_1160616[::-1]],
        dims=["y", "x"],
        coords={"y": range(10), "x": range(9)},
    )


# --- the defaults are the C-Mod deck's settings ---


def test_defaults_are_the_c_mod_decks_settings():
    """Value and Python type of every default, as the twodca deck's C-Mod settings have
    them (fusion_scripts `twodca_manuscript/datasets/cmod.py` at 12b8543). ``stamp``
    writes them with repr, so an int that became a float would show here, and so would
    a field renamed or moved."""
    assert (
        decorrelation.stamp(decorrelation.Averages())
        == "threshold=2.5; window=60; check_max=1; single_counting=True"
    )
    assert decorrelation.stamp(decorrelation.Tracking()) == (
        "position_filter=PositionFilterParams(window_size=1, window_type='hann', "
        "mask_distance=2, mask_signal_factor=0.65, require_within_boundaries=False); "
        "cross_corr_mask_signal_factor=0.75; velocity=VelocityParams(estimator='lsq'); "
        "neighbour_step=1"
    )
    assert (
        decorrelation.stamp(decorrelation.Tde())
        == "min_cc=0.5; running_mean=True; interpolate=True"
    )
    assert decorrelation.stamp(decorrelation.Blobs()) == (
        "gauss_fit=GaussFitParams(size_penalty=5, aspect_penalty=0.2, tilt_penalty=0.2, "
        "size_max=None); taud_estimation=TaudEstimationParams(cutoff=1000000.0, "
        "nperseg=2000.0)"
    )
    assert (
        decorrelation.MIN_CC == 0.5
        and decorrelation.Tde().min_cc is decorrelation.MIN_CC
    )


def test_defaults_are_the_apd_defaults_with_the_decks_changes():
    """The deck starts from the APD defaults and changes five knobs; it sets the
    cross-correlation's mask signal factor apart."""
    deck = get_default_apd_method_params()
    deck.two_dca.threshold = 2.5
    deck.two_dca.window = 60
    deck.position_filter.mask_signal_factor = 0.65
    deck.velocity.estimator = "lsq"
    deck.position_filter.window_size = 1

    averages = decorrelation.Averages()
    for f in dataclasses.fields(decorrelation.Averages):
        assert repr(getattr(averages, f.name)) == repr(getattr(deck.two_dca, f.name))
    tracking = decorrelation.Tracking()
    assert repr(tracking.position_filter) == repr(deck.position_filter)
    assert tracking.position_filter_for("cond_av") is tracking.position_filter
    assert tracking.position_filter_for("cross_corr") == dataclasses.replace(
        deck.position_filter, mask_signal_factor=0.75
    )
    assert repr(tracking.velocity) == repr(deck.velocity)
    blobs = decorrelation.Blobs()
    assert repr(blobs.gauss_fit) == repr(deck.gauss_fit)
    assert repr(blobs.taud_estimation) == repr(deck.taud_estimation)


def test_averages_are_two_dca_params_without_the_pixel():
    names = [f.name for f in dataclasses.fields(TwoDcaParams)]
    assert [f.name for f in dataclasses.fields(decorrelation.Averages)] == [
        n for n in names if n not in ("refx", "refy")
    ]
    averages = decorrelation.Averages(threshold=3.0, window=11)
    two_dca = averages.two_dca(5, 4)
    assert two_dca == TwoDcaParams(
        refx=5, refy=4, threshold=3.0, window=11, check_max=1, single_counting=True
    )
    assert decorrelation.stamp(two_dca) == decorrelation.stamp(averages)


def test_defaults_are_fresh_objects():
    a, b = decorrelation.Tracking(), decorrelation.Tracking()
    a.position_filter.mask_signal_factor = 0.1
    assert b.position_filter.mask_signal_factor == 0.65
    assert decorrelation.Blobs().gauss_fit is not decorrelation.Blobs().gauss_fit


# --- every leaf is one a parameter hash can read ---


def _leaves(cls, prefix=""):
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        path = f"{prefix}{f.name}"
        annotation = hints[f.name]
        if dataclasses.is_dataclass(annotation):
            yield from _leaves(annotation, path + ".")
        else:
            yield path, annotation


@pytest.mark.parametrize(
    "cls",
    [
        decorrelation.Averages,
        decorrelation.Tracking,
        decorrelation.Tde,
        decorrelation.Blobs,
    ],
)
def test_every_leaf_is_hashable(cls):
    """An int, float, str, bool or Enum at every leaf, and no reference pixel: Shot
    Explorer keys its cache on these settings, field by field."""
    leaves = dict(_leaves(cls))
    assert leaves
    for path, annotation in leaves.items():
        assert annotation in (int, float, str, bool) or (
            isinstance(annotation, type) and issubclass(annotation, Enum)
        ), f"{path}: {annotation}"
        assert path.split(".")[-1] not in ("refx", "refy"), path


# --- what importing the package loads, and that the pipeline prints nothing ---


def test_importing_the_package_loads_netcdf4_before_h5py_and_no_exploration_code():
    """netCDF4 first, or every netCDF read fails with Errno -101 once h5py is in; and
    nothing of fusion_scripts, its configuration or the discharge database."""
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, imaging_methods; m = list(sys.modules); "
            "print('h5py' not in m or m.index('netCDF4') < m.index('h5py')); "
            "print(sorted(k for k in m if k.split('.')[0] in ('config', 'density_scan', "
            "'decorrelation', 'twodca_manuscript', 'plotting_scripts', "
            "'waveform_analysis', 'experimental_database', 'figure_provenance', "
            "'seaborn', 'cosmoplots')))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.splitlines()
    assert loaded == ["True", "[]"]


def test_no_output(record, settings, dead, capfd):
    """On a 2 x 2 corner of the record: nothing printed, by any step."""
    corner = record.isel(x=slice(0, 2), y=slice(0, 2))
    corner_dead = dead.isel(x=slice(0, 2), y=slice(0, 2))
    averages, tracking, tde, blob_settings = settings
    found = {
        (x, y): decorrelation.average(corner, averages, x, y)
        for x, y in decorrelation.references(corner_dead.values)
    }
    small = decorrelation.stack(found, corner, averages, corner_dead)
    average_at = lambda x, y: decorrelation.at(small, x, y)  # noqa: E731
    decorrelation.fields(corner, average_at, averages, tracking, tde, corner_dead)
    decorrelation.blobs(corner, average_at, averages, 1, blob_settings, corner_dead)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


# --- the dead-pixel mask ---


def test_dead_mask_stored_in_the_file(record):
    dead = decorrelation.dead_mask(record)
    assert dead.dims == ("y", "x") and dead.dtype == bool and dead.name == "dead"
    assert np.array_equal(dead.values, record.dead.values)
    assert dead.attrs["dead_mask_source"] == record.attrs["dead_mask_source"]
    # A hand-made mask is consulted only where the file stores none.
    other = {
        "0": xr.DataArray(np.ones((fixture.NY, fixture.NX), bool), dims=("y", "x"))
    }
    assert decorrelation.dead_mask(record, hand_made=other).identical(dead)


def test_dead_mask_hand_made_for_its_run_day_only():
    mask = hand_made_1160616()
    blank = xr.Dataset(
        {"frames": (("y", "x", "time"), np.zeros(mask.shape + (3,)))},
        attrs={"shot_number": 1160616027},
    )
    hand_made = {"1160616": mask}
    dead = decorrelation.dead_mask(blank, hand_made=hand_made)
    assert dead.dims == ("y", "x") and dead.dtype == bool
    assert np.array_equal(dead.values, mask.values) and int(dead.sum()) == 22
    assert dead.attrs["dead_mask_source"] == "hand-made, 1160616"
    assert dead.identical(decorrelation.dead_mask(blank, {"1160616": mask.T}))
    # Without the table it raises, as for a day the table does not hold.
    with pytest.raises(ValueError, match="preprocess it again"):
        decorrelation.dead_mask(blank)
    with pytest.raises(ValueError, match="preprocess it again"):
        decorrelation.dead_mask(
            blank.assign_attrs(shot_number=1140827001), hand_made=hand_made
        )
    with pytest.raises(ValueError, match="preprocess it again"):
        decorrelation.dead_mask(blank.assign_attrs(shot_number="none"), hand_made)
    with pytest.raises(ValueError, match="does not match"):
        decorrelation.dead_mask(blank.isel(x=slice(0, 3)), hand_made=hand_made)


# --- average, stack and at ---


@pytest.mark.parametrize("pixel", [(2, 1), (0, 4), fixture.FLAT])
def test_average_is_find_events_and_2dcas(record, settings, pixel):
    averages = settings[0]
    two_dca = TwoDcaParams(
        refx=pixel[0],
        refy=pixel[1],
        threshold=averages.threshold,
        window=averages.window,
        check_max=averages.check_max,
        single_counting=averages.single_counting,
    )
    _, expected = find_events_and_2dca(record, two_dca, verbose=False)
    got = decorrelation.average(record, averages, *pixel)
    assert got.identical(expected)
    if pixel == fixture.FLAT:
        assert got.identical(xr.Dataset())  # no event: the empty average
    else:
        assert got.cond_av.dtype == got.cross_corr.dtype == np.float64


def test_bank_schema(bank, record, dead):
    ny, nx = record.sizes["y"], record.sizes["x"]
    lags = 11  # a window of 10, widened to odd
    assert dict(bank.sizes) == {
        "ref_y": ny,
        "ref_x": nx,
        "y": ny,
        "x": nx,
        "time": lags,
    }
    for v in decorrelation.AVERAGE_VARIABLES:
        assert bank[v].dims == ("ref_y", "ref_x", "y", "x", "time")
        assert bank[v].dtype == np.float64
    assert bank.nevents.dims == bank.computed.dims == ("ref_y", "ref_x")
    assert bank.nevents.dtype == np.int64
    assert bank.computed.dtype == bool and bank.dead.dtype == bool
    assert np.array_equal(bank.computed.values, ~dead.values)
    assert np.array_equal(bank.dead.values, dead.values)
    assert (
        bank.attrs["averages"]
        == "threshold=2.0; window=10; check_max=1; single_counting=True"
    )
    assert bank.attrs["dead_mask_source"] == dead.attrs["dead_mask_source"]
    x, y = fixture.FLAT
    assert bank.nevents.values[y, x] == 0 and bank.computed.values[y, x]
    assert np.isnan(bank.cond_av.values[y, x]).all()
    for x, y in fixture.DEAD:
        assert bank.nevents.values[y, x] == 0 and not bank.computed.values[y, x]


def test_at_gives_every_average_back(bank, fresh, tmp_path):
    for (x, y), average_ds in fresh.items():
        assert decorrelation.at(bank, x, y).identical(average_ds)
    for x, y in fixture.DEAD:
        assert decorrelation.at(bank, x, y) is None
    path = tmp_path / "bank.nc"
    bank.to_netcdf(path)
    with xr.open_dataset(path) as stored:  # lazily, as a store reads it
        for (x, y), average_ds in fresh.items():
            assert decorrelation.at(stored, x, y).identical(average_ds)


def test_stack_refuses_an_incomplete_or_mixed_bank(record, settings, dead, fresh):
    averages = settings[0]
    some = dict(fresh)
    del some[(2, 1)]
    with pytest.raises(ValueError, match="no average"):
        decorrelation.stack(some, record, averages, dead)
    other = dict(fresh)
    other[(2, 1)] = decorrelation.average(
        record, dataclasses.replace(averages, window=12), 2, 1
    )
    with pytest.raises(ValueError, match="lags"):
        decorrelation.stack(other, record, averages, dead)
    # Averages at dead references are left out of the bank.
    extra = {
        **fresh,
        **{p: decorrelation.average(record, averages, *p) for p in fixture.DEAD},
    }
    assert decorrelation.stack(extra, record, averages, dead).identical(
        decorrelation.stack(fresh, record, averages, dead)
    )


# --- the products against the pre-refactor code ---


def test_the_record_is_make_records(record):
    """The stored record is what pipeline_fixture.make_record makes. The frames are
    float32 sums of exponentials, so a platform whose exp rounds differently may flip
    their last bit."""
    made = fixture.make_record()
    stored = xr.load_dataset(fixture.RECORD)
    assert made.drop_vars("frames").identical(stored.drop_vars("frames"))
    assert made.frames.dtype == stored.frames.dtype == np.float32
    np.testing.assert_allclose(
        made.frames.values, stored.frames.values, rtol=0, atol=1e-6
    )
    assert record.frames.dtype == np.float64


def test_fields_equal_the_pre_refactor_code(fields_out):
    expected = xr.load_dataset(fixture.EXPECTED_FIELDS)
    fixture.assert_same(fields_out, expected, list(expected.data_vars))
    assert fields_out.R.identical(expected.R) and fields_out.Z.identical(expected.Z)
    assert np.array_equal(fields_out.time.values, expected.time.values)


def test_fields_schema(fields_out, dead):
    lags = fields_out.sizes["time"]
    assert lags == 11
    for suffix in decorrelation.TRACKS:
        for k in ("vr", "vz", "nlags", "level"):
            assert fields_out[f"{k}_{suffix}"].dims == ("y", "x")
            assert fields_out[f"{k}_{suffix}"].dtype == np.float64
        for k in ("pos_r", "pos_z", "fit"):
            assert fields_out[f"{k}_{suffix}"].dims == ("y", "x", "time")
        assert fields_out[f"fit_{suffix}"].dtype == bool
        live = ~dead.values & (fields_out.nevents.values > 0)
        assert np.array_equal(
            fields_out[f"nlags_{suffix}"].values[live],
            fields_out[f"fit_{suffix}"].values.sum(-1)[live],
        )
    assert (
        np.isnan(fields_out.level_max).all() and np.isnan(fields_out.level_2dcc).all()
    )
    names = {f"{k}_tde" for k in decorrelation.TDE_VARIABLES} | {
        f"{k}_catde" for k in decorrelation.CATDE_VARIABLES
    }
    for k in names | {"nevents"}:
        assert fields_out[k].dims == ("y", "x") and fields_out[k].dtype == np.float64
    assert fields_out.dead.dtype == bool
    assert np.array_equal(fields_out.dead.values, dead.values)
    per_pixel = [
        k
        for k in fields_out.data_vars
        if fields_out[k].dims == ("y", "x") and k != "dead"
    ]
    assert np.isnan(fields_out[per_pixel].to_array().values[:, dead.values]).all()
    assert set(fields_out.attrs) == {
        "min_cc",
        "averages",
        "tracking",
        "tde",
        "dead_mask_source",
    }


def test_blobs_equal_the_pre_refactor_code(blobs_out, dead):
    expected = xr.load_dataset(fixture.EXPECTED_BLOBS)
    fixture.assert_same(blobs_out, expected, list(expected.data_vars))
    assert list(decorrelation.BLOB_PARAMETERS) == list(expected.data_vars)
    assert blobs_out.dead.dtype == bool
    assert np.array_equal(blobs_out.dead.values, dead.values)
    assert blobs_out.attrs["units"] == "m, m^2, s, rad"
    assert blobs_out.attrs["blobs"].startswith(
        "gauss_fit=GaussFitParams(size_penalty=5, aspect_penalty=0.3"
    )


def test_fresh_and_stored_averages_give_identical_products(
    record, settings, dead, fresh, bank, tmp_path
):
    """No casting, end to end: fields straight after the 2DCA, and later off the bank
    read from disk."""
    averages, tracking, tde, _ = settings
    pixels = [(2, 1), (2, 3), (0, 4), fixture.FLAT]
    straight = decorrelation.fields(
        record,
        lambda x, y: fresh[(x, y)],
        averages,
        tracking,
        tde,
        dead,
        pixels=pixels,
    )
    path = tmp_path / "bank.nc"
    bank.to_netcdf(path)
    with xr.open_dataset(path) as stored:
        later = decorrelation.fields(
            record,
            lambda x, y: decorrelation.at(stored, x, y),
            averages,
            tracking,
            tde,
            dead,
            pixels=pixels,
        )
    assert later.identical(straight)


def test_fields_at_some_pixels_leave_the_rest_nan(
    record, settings, dead, fresh, fields_out
):
    averages, tracking, tde, _ = settings
    pixels = [(2, 1), (1, 2), (3, 3)]  # (1, 2) is dead
    some = decorrelation.fields(
        record, lambda x, y: fresh[(x, y)], averages, tracking, tde, dead, pixels=pixels
    )
    for x, y in [(2, 1), (3, 3)]:
        # Tracks are per reference; the TDE shares delays across pixels.
        for suffix in decorrelation.TRACKS:
            for k in ("vr", "vz", "nlags", "level"):
                a = some[f"{k}_{suffix}"].values[y, x]
                b = fields_out[f"{k}_{suffix}"].values[y, x]
                assert a == b or (np.isnan(a) and np.isnan(b))
    elsewhere = np.ones(dead.shape, bool)
    elsewhere[1, 2] = elsewhere[3, 3] = False
    assert np.isnan(some.vr_com.values[elsewhere]).all()
    assert np.isnan(some.vr3_tde.values[elsewhere]).all()


def test_track_is_one_pixel_of_the_fields(fresh, settings, fields_out):
    """``track`` with the parameters ``fields`` builds for the centroid track: the
    tracking settings and the pixel's own contour level."""
    tracking = settings[1]
    x, y = 2, 1
    level = neighbour_level(fresh[(x, y)], "cond_av", tracking.neighbour_step)
    params = MethodParameters(
        position_filter=tracking.position_filter_for("cond_av"),
        velocity=tracking.velocity,
        contouring=ContouringParams(threshold_factor=level),
    )
    vr, vz, position, fit = decorrelation.track(
        fresh[(x, y)], "cond_av", "contouring", params
    )
    assert np.isfinite(vr) and vr == fields_out.vr_com.values[y, x]
    assert vz == fields_out.vz_com.values[y, x]
    np.testing.assert_array_equal(position[:, 0], fields_out.pos_r_com.values[y, x])
    np.testing.assert_array_equal(position[:, 1], fields_out.pos_z_com.values[y, x])
    np.testing.assert_array_equal(fit, fields_out.fit_com.values[y, x])


# --- stamp, nevents and the paper's cut ---


def test_stamp_reads_every_field_but_the_pixel():
    assert (
        decorrelation.stamp(TwoDcaParams(refx=3, refy=7))
        == "threshold=2.5; window=60; check_max=1; single_counting=True"
    )
    assert decorrelation.stamp(decorrelation.Tde(min_cc=0.25, interpolate=False)) == (
        "min_cc=0.25; running_mean=True; interpolate=False"
    )
    with pytest.raises(TypeError):
        decorrelation.stamp({"threshold": 2.5})  # a dataclass, nothing else


def test_nevents(fresh):
    assert decorrelation.nevents(xr.Dataset()) == 0
    assert decorrelation.nevents(fresh[fixture.FLAT]) == 0
    average_ds = fresh[(2, 1)]
    count = decorrelation.nevents(average_ds)
    assert type(count) is int and count == int(average_ds.number_events) > 0


def _paper_fields(ny=4, nx=5):
    """A ``fields`` Dataset that passes the paper's cut everywhere but at the border."""
    full = lambda value: (("y", "x"), np.full((ny, nx), value))  # noqa: E731
    names = ("vr_com", "vz_com", "vr3_tde", "vz3_tde", "vr2_tde", "vz2_tde")
    return xr.Dataset(
        {
            "nlags_com": full(float(decorrelation.MIN_LAGS)),
            "nevents": full(float(decorrelation.MIN_EVENTS)),
            **{v: full(500.0) for v in names},
            "nlags_max": full(0.0),
            "vr_max": full(np.nan),
            "vz_max": full(np.nan),
        }
    )


def test_reliable_keeps_the_interior_where_every_method_has_a_number():
    f = _paper_fields()
    ok = decorrelation.reliable(f)
    assert ok.dims == ("y", "x") and ok.dtype == bool
    interior = np.zeros((4, 5), bool)
    interior[1:-1, 1:-1] = True
    assert np.array_equal(ok.values, interior)
    # The maximum track has neither lags nor velocities here.
    assert not decorrelation.reliable(f, track="_max").values.any()


@pytest.mark.parametrize(
    "variable, value",
    [
        ("nlags_com", decorrelation.MIN_LAGS - 1),
        ("nevents", decorrelation.MIN_EVENTS - 1),
        ("nevents", np.nan),
        ("vr_com", np.nan),
        ("vz_com", np.nan),
        ("vr3_tde", np.nan),
        ("vz3_tde", np.inf),
        ("vr2_tde", np.nan),
        ("vz2_tde", np.nan),
    ],
)
def test_reliable_drops_a_pixel_short_of_any_condition(variable, value):
    f = _paper_fields()
    f[variable].values[1, 2] = value
    ok = decorrelation.reliable(f).values
    assert not ok[1, 2] and ok[2, 2] and ok[1, 1]


def test_reliable_with_events_or_a_further_estimate():
    f = _paper_fields()
    f["nevents"].values[1, 2] = 10
    assert decorrelation.reliable(f, min_events=10).values[1, 2]
    assert not decorrelation.reliable(f).values[1, 2]
    ca = xr.Dataset({"vr": (("y", "x"), np.ones((4, 5)))})
    ca["vr"].values[2, 3] = np.nan
    assert decorrelation.reliable(_paper_fields()).values[2, 3]
    assert not decorrelation.reliable(_paper_fields(), ca=ca).values[2, 3]


def test_reliable_on_the_fixtures_fields(fields_out):
    ok = decorrelation.reliable(fields_out, min_events=0).values
    f = fields_out
    expected = (f.nlags_com.values >= decorrelation.MIN_LAGS) & (f.nevents.values >= 0)
    for v in ("vr_com", "vz_com", "vr3_tde", "vz3_tde", "vr2_tde", "vz2_tde"):
        expected &= np.isfinite(f[v].values)
    expected[[0, -1], :] = expected[:, [0, -1]] = False
    assert np.array_equal(ok, expected)


def _reliable_as_moved(
    f, ca=None, min_events=decorrelation.MIN_EVENTS, track=decorrelation.TRACK
):
    """reliable() as it was moved from fusion_scripts' figures.py, before min_lags: the
    lag threshold is the module's MIN_LAGS."""
    ok = (f["nlags" + track] >= decorrelation.MIN_LAGS) & (f.nevents >= min_events)
    for v in ("vr" + track, "vz" + track, "vr3_tde", "vz3_tde", "vr2_tde", "vz2_tde"):
        ok &= np.isfinite(f[v])
    if ca is not None:
        ok &= np.isfinite(ca.vr)
    ny, nx = ok.shape
    interior = np.zeros((ny, nx), bool)
    interior[1:-1, 1:-1] = True
    return ok & interior


def test_reliable_min_lags_moves_the_cut_and_its_default_is_the_old_one(fields_out):
    paper = _paper_fields()
    paper["nlags_com"].values[1, 2] = 5
    ca = xr.Dataset({"vr": fields_out.vr3_catde})
    # The default is the cut as it was moved, whatever else is passed.
    for f, kwargs in [
        (paper, {}),
        (paper, {"min_events": 0, "track": "_max"}),
        (fields_out, {"min_events": 0}),
        (fields_out, {"ca": ca, "min_events": 0}),
        (fields_out, {"min_events": 0, "track": "_2dcc"}),
    ]:
        expected = _reliable_as_moved(f, **kwargs)
        assert decorrelation.reliable(f, **kwargs).identical(expected)
        assert decorrelation.reliable(
            f, min_lags=decorrelation.MIN_LAGS, **kwargs
        ).identical(expected)
    # The caller's own threshold moves it: the pixel whose fit rests on 5 lags is in at
    # 5 and out at the paper's 8, and 9 leaves none.
    assert not decorrelation.reliable(paper).values[1, 2]
    assert decorrelation.reliable(paper, min_lags=5).values[1, 2]
    assert not decorrelation.reliable(paper, min_lags=9).values.any()
    # Keyword-only: a call written for the signature without min_lags,
    # reliable(f, None, 100), would otherwise set min_lags to 100.
    with pytest.raises(TypeError):
        decorrelation.reliable(paper, None, 100)
    # On the fixture's fields the centroid fits rest on 3 to 6 of their 11 lags, so the
    # paper's 8 keeps none, and 4 keeps the interior pixels that reach it.
    assert not decorrelation.reliable(fields_out, min_events=0).values.any()
    kept = decorrelation.reliable(fields_out, min_lags=4, min_events=0).values
    assert kept.any() and (fields_out.nlags_com.values[kept] >= 4).all()

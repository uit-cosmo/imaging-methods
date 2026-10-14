"""Every method's velocity field and the blob parameters at every pixel of one record.

Moved from fusion_scripts `decorrelation/pipeline.py` at 12b8543, with ``stamp`` and
``nevents`` from `twodca_manuscript/average_cache.py`, ``MIN_CC`` from
`twodca_manuscript/tde_field.py` and ``reliable`` from `decorrelation/apd_check/figures.py`.

This is the one implementation: the paper's multi-shot runs and Shot Explorer's
``pixel_averages``, ``method_fields`` and ``blob_parameters`` products call it, so the
paper and the app read the same numbers.

The 2DCA is over 95% of the cost and depends on ``Averages`` alone, so it is computed
once per record and settings, and every product reads it back through
``average_at(x, y)``::

    ds  the preprocessed record: frames (y, x, time) float64 in memory, R and Z in metres
     |  average(ds, averages, x, y) at every live reference, then stack()
     v
    bank (pixel_averages): cond_av, cond_repr, cross_corr for every reference
     |  average_at = lambda x, y: at(bank, x, y)
     v
    fields(ds, average_at, ...)  every method's v_R, v_Z and its track     (method_fields)
    blobs(ds, average_at, ...)   the 13 blob parameters                    (blob_parameters)

``average_at`` can as well read averages kept anywhere else (the paper keeps one file
per reference): a product computed from a fresh average and one computed from the
stored bank are bit-identical, since nothing on the way is cast.

Pure: arguments in, ``xr.Dataset`` out. No caching, no paths, no printing, and no
plotting. Warnings are silenced inside the loops, as the paper's scripts always ran
them.

The defaults are the C-Mod settings of the paper's twodca deck, as literal values (see
``Averages``): the same for every shot, so a shot they do not suit shows up as failed
pixels rather than being retuned. Nothing else assumes C-Mod: array shapes, pixel
pitches and sampling are read off the data, and a run day's hand-made dead-pixel mask
is an argument of ``dead_mask``.
"""

import dataclasses
import warnings
from dataclasses import dataclass, field, replace

import netCDF4  # noqa: F401  -- before h5py, which velocity_estimation imports, or every netCDF read fails
import numpy as np
import xarray as xr
import velocity_estimation as ve

from .cond_av import find_events_and_2dca
from .contours import get_contour_evolution, neighbour_level
from .duration_time_estimation import (
    Analytics,
    DurationTimeEstimator,
    SecondOrderStatistic,
)
from .maximum_trajectory import get_maximum_time
from .method_parameters import (
    ContouringParams,
    GaussFitParams,
    MethodParameters,
    PositionFilterParams,
    TaudEstimationParams,
    TwoDcaParams,
    VelocityParams,
)
from .parameter_estimation import estimate_fwhm_sizes, fit_ellipse_to_event
from .utils import MaskedImagingDataInterface, get_dt
from .velocity_estimates import (
    get_averaged_velocity_from_position,
    get_positions_and_mask,
)

__all__ = [
    "Averages",
    "Tracking",
    "Tde",
    "Blobs",
    "TRACKS",
    "AVERAGE_VARIABLES",
    "TDE_VARIABLES",
    "CATDE_VARIABLES",
    "BLOB_PARAMETERS",
    "BLOB_UNITS",
    "MIN_CC",
    "MIN_LAGS",
    "MIN_EVENTS",
    "TRACK",
    "dead_mask",
    "references",
    "nevents",
    "stamp",
    "average",
    "stack",
    "at",
    "track",
    "tde_fields",
    "ca_tde",
    "fields",
    "pixel_blob",
    "blobs",
    "reliable",
]

# The three 2DCA tracks, by the suffix of their variables: the conditional average by
# its subpixel maximum and by the centroid of its contour, and the cross-correlation by
# its maximum.
TRACKS = {
    "max": ("cond_av", "max"),
    "com": ("cond_av", "contouring"),
    "2dcc": ("cross_corr", "max"),
}
AVERAGE_VARIABLES = ("cond_av", "cond_repr", "cross_corr")
TDE_VARIABLES = ("vr3", "vz3", "vr2", "vz2", "cc")  # off the record, suffixed _tde
CATDE_VARIABLES = ("vr3", "vz3", "vr2", "vz2")  # on the conditional average, _catde
BLOB_PARAMETERS = (
    "nevents",
    "level",
    "area",
    "lx_c",
    "ly_c",
    "theta_c",
    "lr",
    "lz",
    "lx_f",
    "ly_f",
    "theta_f",
    "taud",
    "lam",
)
BLOB_UNITS = "m, m^2, s, rad"

# Fewest the CCF between two neighbours may peak at for the delay between them to be
# believed. It is the estimator's own floor, not a cut applied afterwards: a pixel whose
# neighbours correlate less than this comes back without a velocity at all. 0.5 is what
# the deck's single-puff figure has always used.
MIN_CC = 0.5

# The paper's cut (``reliable``): the fewest lags a 2DCA fit may rest on, and the fewest
# events behind the average.
MIN_LAGS, MIN_EVENTS = 8, 200
# The 2DCA track ``reliable`` reads: the centroid. The subpixel maximum sticks to the
# reference pixel for a few lags where the pulse passes close to its centre (pixel
# (5, 7) of 1160616027 reads 166 m/s by the maximum against 603 m/s by the centroid).
TRACK = "_com"


# ---------------------------------------------------------------------------------------
# Settings
#
# Every leaf is an int, float, str or bool, or an imaging_methods dataclass of those, so
# Shot Explorer hashes them as they are. No reference pixel (it would mint one cache key
# per pixel for one result), no view cut, no tuple or dict.
#
# The defaults are the C-Mod settings of the twodca deck, copied as literal values from
# fusion_scripts `twodca_manuscript/datasets/cmod.py` at 12b8543: its ``params()``,
# which starts from ``get_default_apd_method_params()`` and sets the 2DCA threshold to
# 2.5 and its window to 60, the mask signal factor to 0.65, the position smoothing
# window to 1 and the velocity estimator to "lsq"; its ``params_by_variable``, which
# sets the cross-correlation's mask signal factor to 0.75; and the ``DatasetSpec``
# default ``neighbour_step`` of 1. Every value keeps the Python type it has there (the
# Gaussian fit's size_penalty is the int 5), since ``stamp`` writes them with repr and
# Shot Explorer's cache keys hash them.
# ---------------------------------------------------------------------------------------


@dataclass
class Averages:
    """What one reference's 2DCA depends on, bar the pixel: ``TwoDcaParams`` without
    refx and refy, field for field and in its order, so ``stamp`` reads the same off
    both.

    threshold: the reference must exceed it, in standard deviations of the normalized
        record.
    window: samples kept around each event's peak; an even window is widened to the
        next odd one.
    check_max: radius of the neighbourhood the reference must be the maximum of at the
        peak; 0 is off.
    single_counting: keep events at least a window apart, the highest first.
    """

    threshold: float = 2.5
    window: int = 60
    check_max: int = 1
    single_counting: bool = True

    def two_dca(self, x, y):
        """The 2DCA's parameters with (x, y) as the reference."""
        return TwoDcaParams(
            refx=x,
            refy=y,
            threshold=self.threshold,
            window=self.window,
            check_max=self.check_max,
            single_counting=self.single_counting,
        )


@dataclass
class Tracking:
    """How the three 2DCA tracks are read off an average.

    position_filter: cond_av's mask and smoothing.
    cross_corr_mask_signal_factor: cross_corr's mask signal factor, its one difference
        from cond_av's filter: the cross-correlation sits on a pedestal, so a factor
        chosen for the decaying average cannot bind on it (the deck's
        ``params_by_variable``, flattened).
    velocity: how a velocity is reduced from the tracked position; "lsq" is the
        least-squares slope.
    neighbour_step: pixels between the peak and the neighbours the contour level is read
        off (``contours.neighbour_level``). The level itself is read off each average,
        not set.
    """

    position_filter: PositionFilterParams = field(
        default_factory=lambda: PositionFilterParams(
            window_size=1,
            window_type="hann",
            mask_distance=2,
            mask_signal_factor=0.65,
            require_within_boundaries=False,
        )
    )
    cross_corr_mask_signal_factor: float = 0.75
    velocity: VelocityParams = field(
        default_factory=lambda: VelocityParams(estimator="lsq")
    )
    neighbour_step: int = 1

    def position_filter_for(self, variable):
        """The position filter ``variable``'s track runs with."""
        if variable == "cross_corr":
            return replace(
                self.position_filter,
                mask_signal_factor=self.cross_corr_mask_signal_factor,
            )
        return self.position_filter


@dataclass
class Tde:
    """Time delay estimation off the record's cross-correlation (velocity_estimation,
    TDEMethod.CC).

    min_cc: the smallest cross-correlation peak a delay is believed at (``MIN_CC``).
    running_mean: smooth the cross-correlation until it has a single local maximum.
    interpolate: read the delay off an interpolated peak. On here, unlike the deck's
        TDE field: the prediction under test is about the lag of the maximum itself,
        and at 0.5 us sampling a poloidal delay of a few samples otherwise leaves the
        two-point v_Z quantized.
    """

    min_cc: float = MIN_CC
    running_mean: bool = True
    interpolate: bool = True


@dataclass
class Blobs:
    """The blob parameters' own settings: the penalised ellipse fit and the PSD
    duration-time fit."""

    gauss_fit: GaussFitParams = field(
        default_factory=lambda: GaussFitParams(
            size_penalty=5, aspect_penalty=0.2, tilt_penalty=0.2, size_max=None
        )
    )
    taud_estimation: TaudEstimationParams = field(
        default_factory=lambda: TaudEstimationParams(cutoff=1e6, nperseg=2e3)
    )


# ---------------------------------------------------------------------------------------
# Dead pixels
# ---------------------------------------------------------------------------------------


def dead_mask(ds, hand_made=None):
    """The dead pixels of the record ``ds``: bool (y, x), its source in
    ``attrs["dead_mask_source"]``.

    - ``ds["dead"]``, when the preprocessed file stores the mask it was made with;
    - otherwise the hand-made mask of the shot's run day (the first seven digits of
      ``ds.attrs["shot_number"]``, a C-Mod shot number ``1YYMMDDnnn``), when
      ``hand_made`` holds one. ``hand_made`` maps a run day, as a 7-digit str
      ("1160616"), to that day's mask, a bool DataArray over y and x, as
      ``experimental_database.hand_made_dead_pixel_masks()`` gives them;
    - otherwise it raises: such a file was preprocessed before masks were stored,
      possibly with another run day's mask (the 1140827 files carry the 2016 one), and
      must be preprocessed again.
    """
    shape = (ds.sizes["y"], ds.sizes["x"])
    if "dead" in ds:
        values = ds["dead"].transpose("y", "x").values.astype(bool)
        source = str(
            ds.attrs.get("dead_mask_source", "stored in the preprocessed file")
        )
    else:
        shot = str(ds.attrs.get("shot_number", ""))
        mask = (hand_made or {}).get(shot[:7]) if len(shot) == 10 else None
        if mask is None:
            raise ValueError(
                f"shot {shot or '(no shot_number)'}: the preprocessed file stores no "
                f"dead-pixel mask, and there is no hand-made one for its run day. It "
                f"was preprocessed before masks were stored, perhaps with another "
                f"day's mask: preprocess it again."
            )
        values = mask.transpose("y", "x").values.astype(bool)
        source = f"hand-made, {shot[:7]}"
    if values.shape != shape:
        raise ValueError(
            f"dead-pixel mask {values.shape} does not match the record's (y, x) {shape}"
        )
    return xr.DataArray(
        values, dims=("y", "x"), name="dead", attrs={"dead_mask_source": source}
    )


def _dead(dead, ds):
    """``dead`` (a DataArray over y and x, or an array) as a (y, x) bool array of
    ``ds``'s shape, and the source it records."""
    if isinstance(dead, xr.DataArray):
        values = dead.transpose("y", "x").values
        source = str(dead.attrs.get("dead_mask_source", "unspecified"))
    else:
        values, source = np.asarray(dead), "unspecified"
    values = values.astype(bool)
    shape = (ds.sizes["y"], ds.sizes["x"])
    if values.shape != shape:
        raise ValueError(
            f"dead-pixel mask {values.shape} does not match the record's (y, x) {shape}"
        )
    return values, source


def references(dead, pixels=None):
    """The live references, as (x, y): ``pixels`` in the order given, every pixel in
    reading order by default, bar the dead ones in ``dead`` (a (y, x) bool array)."""
    dead = np.asarray(dead, dtype=bool)
    if pixels is None:
        ny, nx = dead.shape
        pixels = [(x, y) for y in range(ny) for x in range(nx)]
    return [(int(x), int(y)) for x, y in pixels if not dead[y, x]]


# ---------------------------------------------------------------------------------------
# The 2DCA, and the bank of every reference's average
# ---------------------------------------------------------------------------------------


def nevents(average_ds):
    """Events behind an average; 0 for the empty one a reference without events gets.

    The average carries the count (``number_events``), so the events themselves need
    not be kept.
    """
    return int(average_ds.number_events) if "number_events" in average_ds else 0


def stamp(two_dca):
    """The settings ``two_dca`` (a dataclass) holds, bar the reference pixel, as one
    string: ``name=repr(value)`` for every field but refx and refy, in field order,
    joined by "; ".

    Compared verbatim, so it only has to be stable and complete, not readable -- though
    being readable is what lets a stale file be diagnosed with ncdump.
    """
    return "; ".join(
        f"{f.name}={getattr(two_dca, f.name)!r}"
        for f in dataclasses.fields(two_dca)
        if f.name not in ("refx", "refy")
    )


def average(ds, averages, x, y):
    """The 2DCA with (x, y) as the reference: ``find_events_and_2dca``'s average as it
    returns it, uncast.

    cond_av, cond_repr and cross_corr (y, x, time) in float64, with refx, refy and
    number_events; the empty Dataset when no event qualifies.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, average_ds = find_events_and_2dca(ds, averages.two_dca(x, y), verbose=False)
    return average_ds


def _window(averages):
    """Lags in an average: the window, widened to odd as find_events_and_2dca widens
    it."""
    return averages.window + 1 - averages.window % 2


def _lags(ds, averages):
    """The lags find_events_and_2dca gives an average of ``ds``, for when no reference
    has one to read them off: its own arithmetic, so equal to the bit."""
    window = _window(averages)
    half = (window - 1) // 2
    dt = float(ds["time"][1].values - ds["time"][0].values)
    return (np.arange(window) - half) * dt


def _common_lags(found, ds, averages):
    """The lag axis of every non-empty average in ``found`` ((x, y) -> average), which
    must agree."""
    lags = None
    for (x, y), average_ds in found.items():
        if lags is None:
            lags = average_ds.time.values
        elif not np.array_equal(average_ds.time.values, lags):
            raise ValueError(f"the average at ({x}, {y}) has other lags than the rest")
    if lags is None:
        return _lags(ds, averages)
    if lags.size != _window(averages):
        raise ValueError(
            f"the averages have {lags.size} lags, and a window of {averages.window} "
            f"gives {_window(averages)}"
        )
    return lags


def _spatial_coords(dataset):
    """``dataset``'s coordinates over y and x alone (R and Z), as plain in-memory
    variables."""
    return {
        k: xr.Variable(c.dims, c.values, c.attrs)
        for k, c in dataset.coords.items()
        if set(c.dims) <= {"y", "x"}
    }


def stack(averages_by_pixel, ds, averages, dead):
    """Every live reference's average in one Dataset: the ``pixel_averages`` bank.

    ``averages_by_pixel`` maps (x, y) to ``average(ds, averages, x, y)`` at every live
    reference; averages at dead references are left out. The bank::

        cond_av, cond_repr, cross_corr  (ref_y, ref_x, y, x, time)  float64, NaN at
                                                                    skipped references
        nevents                         (ref_y, ref_x)  int64, 0 where none or skipped
        computed                        (ref_y, ref_x)  bool, False at dead references
        dead                            (y, x)          bool, the mask used
        attrs: averages (``stamp``), dead_mask_source

    Raises unless ``at`` gives every average back exactly as it was given.
    """
    dead_values, source = _dead(dead, ds)
    ny, nx = dead_values.shape
    live = references(dead_values)
    missing = [p for p in live if p not in averages_by_pixel]
    if missing:
        raise ValueError(f"no average at the live references {missing}")
    found = {p: averages_by_pixel[p] for p in live if nevents(averages_by_pixel[p])}
    for (x, y), average_ds in found.items():
        for v in AVERAGE_VARIABLES:
            if average_ds[v].dims != ("y", "x", "time"):
                raise ValueError(
                    f"{v} at ({x}, {y}) has dims {average_ds[v].dims}: the bank takes "
                    "averages of a record whose frames are (y, x, time)"
                )
    lags = _common_lags(found, ds, averages)
    data = {v: np.full((ny, nx, ny, nx, lags.size), np.nan) for v in AVERAGE_VARIABLES}
    count = np.zeros((ny, nx), np.int64)
    computed = np.zeros((ny, nx), bool)
    for x, y in live:
        computed[y, x] = True
        count[y, x] = nevents(averages_by_pixel[(x, y)])
    for (x, y), average_ds in found.items():
        for v in AVERAGE_VARIABLES:
            data[v][y, x] = average_ds[v].values

    reference = ("ref_y", "ref_x")
    bank = xr.Dataset(
        {
            **{v: (reference + ("y", "x", "time"), data[v]) for v in AVERAGE_VARIABLES},
            "nevents": (reference, count),
            "computed": (reference, computed),
            "dead": (("y", "x"), dead_values),
        },
        coords={
            **_spatial_coords(found[next(iter(found))] if found else ds),
            "time": lags,
        },
        attrs={"averages": stamp(averages), "dead_mask_source": source},
    )
    for x, y in live:
        if not at(bank, x, y).identical(averages_by_pixel[(x, y)]):
            raise ValueError(
                f"the bank does not give the average at ({x}, {y}) back as it was given"
            )
    return bank


def at(bank, x, y):
    """Reference (x, y)'s average out of the bank, as ``average`` made it; None if it
    was not computed (a dead reference). Reads that reference alone from a bank opened
    lazily."""
    if not bool(bank["computed"].values[y, x]):
        return None
    count = int(bank["nevents"].values[y, x])
    if not count:
        return xr.Dataset()
    out = xr.Dataset(
        {
            v: (("y", "x", "time"), bank[v].isel(ref_y=y, ref_x=x).values)
            for v in AVERAGE_VARIABLES
        },
        coords={**_spatial_coords(bank), "time": bank["time"].values},
    )
    out["refx"] = int(x)
    out["refy"] = int(y)
    out["number_events"] = count
    return out


# ---------------------------------------------------------------------------------------
# Velocity fields
# ---------------------------------------------------------------------------------------


def track(average_ds, variable, method, position_filter, estimator, level=None):
    """One track through one reference's average: ``(vr, vz, position, fit)``.

    ``method`` is "max", the subpixel maximum of ``variable`` at each lag, or
    "contouring", the centroid of its contour at ``level`` (a fraction of the average's
    maximum). ``position`` is (time, 2), R and Z at each lag of the average, NaN where
    the tracker found nothing; ``fit`` is (time,) bool, the lags the velocity's slope
    rests on. A lag in ``fit`` without a position is fitted where the tracker
    interpolated, as ``smooth_da`` fills it. The velocities are NaN where the fit fails.
    """
    if method == "contouring":
        if level is None:
            raise ValueError("a contour-tracked track needs its level")
        contouring = ContouringParams(threshold_factor=level)
    else:
        contouring = ContouringParams()
    params = MethodParameters(
        position_filter=position_filter,
        velocity=VelocityParams(estimator=estimator),
        contouring=contouring,
    )
    pos, mask, valid = get_positions_and_mask(
        average_ds, variable, params, position_method=method
    )
    with np.errstate(invalid="ignore"):
        vr, vz = get_averaged_velocity_from_position(
            position_da=pos, mask=mask, estimator=estimator
        )
    lags = average_ds.time.values
    index = np.searchsorted(lags, pos.time.values[0]) + np.arange(pos.time.size)
    if not np.array_equal(lags[index], pos.time.values):
        raise ValueError("the tracked positions are not on the average's lags")
    position = np.full((lags.size, 2), np.nan)
    position[index[valid]] = pos.values[valid]
    fit = np.zeros(lags.size, bool)
    fit[index] = mask
    return vr, vz, position, fit


def tde_fields(ds, tde, dead, pixels=None):
    """Three- and two-point TDE off the record's cross-correlation, sharing the delays.

    At ``pixels`` (every live pixel by default), in that order: the delays are cached by
    pixel pair, and a pair met the other way round reuses its delay negated, so the
    order is part of the result. Returns ``{"vr3", "vz3", "vr2", "vz2", "cc"}``, each
    (y, x), NaN where there is no estimate.
    """
    dead_values, _ = _dead(dead, ds)
    # Fresh options: EstimationOptions() otherwise shares its default CCOptions and
    # NeighbourOptions with every caller in the process, so setting these would change
    # what the others get, and what another caller set (a ccf_min_lag, say) would change
    # this.
    options = ve.CCOptions()
    options.minimum_cc_value = tde.min_cc
    options.running_mean = tde.running_mean
    options.interpolate = tde.interpolate
    eo = ve.EstimationOptions()
    eo.method = ve.TDEMethod.CC
    eo.cc_options = options
    eo.neighbour_options = ve.NeighbourOptions()
    delegator = ve.TDEDelegator(eo.method, eo.get_time_delay_options(), True)
    data = MaskedImagingDataInterface(ds, xr.DataArray(dead_values, dims=("y", "x")))
    out = {k: np.full(dead_values.shape, np.nan) for k in TDE_VARIABLES}
    for x, y in references(dead_values, pixels):
        for three, suffix in ((True, "3"), (False, "2")):
            eo.use_3point_method = three
            try:
                p = ve.estimate_velocities_for_pixel(x, y, data, eo, delegator)
            except ValueError:  # both delays zero: the 3-point system is singular
                continue
            out["vr" + suffix][y, x], out["vz" + suffix][y, x] = p.vx, p.vy
            out["cc"][y, x] = p.confidence if np.isfinite(p.vx) else np.nan
    return out


def _delays(e, x, y, dead):
    """The delays along R and Z off an event's peak times (``get_maximum_time``),
    skipping dead neighbours."""
    t0 = get_maximum_time(e, x, y)
    taus = []
    for pairs in (((x + 1, y), (x - 1, y)), ((x, y + 1), (x, y - 1))):
        t = [
            s * (get_maximum_time(e, *q) - t0)
            for q, s in zip(pairs, (1, -1))
            if 0 <= q[0] < e.sizes["x"]
            and 0 <= q[1] < e.sizes["y"]
            and not dead[q[1], q[0]]
        ]
        taus.append(np.mean(t) if t else np.nan)
    return taus


def ca_tde(average_ds, x, y, dead):
    """Three- and two-point TDE on the conditional average itself: the lag of its
    maximum at the four neighbours, dead ones (a (y, x) bool array) skipped.
    ``(vr3, vz3, vr2, vz2)``, NaN where it fails."""
    e = average_ds.cond_av
    try:
        taux, tauy = _delays(e, x, y, dead)
        dr = float(e.R.isel(x=1, y=0) - e.R.isel(x=0, y=0))
        dz = float(e.Z.isel(x=0, y=1) - e.Z.isel(x=0, y=0))
        three = ve.get_2d_velocities_from_time_delays(taux, tauy, dr, 0, 0, dz)
    except Exception:
        return (np.nan,) * 4
    two = (dr / taux if taux else np.nan, dz / tauy if tauy else np.nan)
    return (*three, *two)


def fields(ds, average_at, averages, tracking, tde, dead, pixels=None):
    """Every method's velocity field over the record ``ds``: the ``method_fields``
    product.

    ``average_at(x, y)`` gives the 2DCA at reference (x, y) under ``averages`` (``at`` on
    a bank, or a cache); it is asked at live references only. ``pixels`` limits the
    references to those; the rest stay NaN. Everything is NaN at dead pixels. The
    Dataset::

        for each track T in max, com, 2dcc (TRACKS):
          vr_T, vz_T, nlags_T      (y, x)        the velocity and the lags its slope
                                                 rests on
          level_T                  (y, x)        the contour level, NaN for a maximum
                                                 track
          pos_r_T, pos_z_T         (y, x, time)  the tracked position, NaN where
                                                 untracked
          fit_T                    (y, x, time)  bool, the lags the slope rests on
        nevents                                  (y, x)
        vr3_tde vz3_tde vr2_tde vz2_tde cc_tde   (y, x)  TDE off the record (CC)
        vr3_catde vz3_catde vr2_catde vz2_catde  (y, x)  TDE on the conditional average
        dead                                     (y, x)  bool, the mask used
        attrs: min_cc, averages, tracking, tde (the settings, stamped), dead_mask_source

    The (y, x) variables are float64, as in the paper's ``apd<shot>_velocities.nc``,
    whose variables these are by name.
    """
    dead_values, source = _dead(dead, ds)
    ny, nx = dead_values.shape
    live = references(dead_values, pixels)
    out = {
        suffix: {
            "vr": np.full((ny, nx), np.nan),
            "vz": np.full((ny, nx), np.nan),
            "nlags": np.zeros((ny, nx), dtype=int),
            "level": np.full((ny, nx), np.nan),
        }
        for suffix in TRACKS
    }
    count = np.zeros((ny, nx), dtype=int)
    catde = np.full((len(CATDE_VARIABLES), ny, nx), np.nan)
    found = {}
    tracked = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y in live:
            average_ds = average_at(x, y)
            if average_ds is None:
                raise ValueError(f"no average at the live reference ({x}, {y})")
            count[y, x] = nevents(average_ds)
            if not count[y, x]:
                continue
            found[(x, y)] = average_ds
            for suffix, (variable, method) in TRACKS.items():
                level = None
                if method == "contouring":
                    # A property of this pixel's average, not a setting: read off it
                    # before tracking.
                    level = neighbour_level(
                        average_ds, variable, tracking.neighbour_step
                    )
                    out[suffix]["level"][y, x] = level
                vr, vz, position, fit = track(
                    average_ds,
                    variable,
                    method,
                    tracking.position_filter_for(variable),
                    tracking.velocity.estimator,
                    level,
                )
                out[suffix]["vr"][y, x], out[suffix]["vz"][y, x] = vr, vz
                out[suffix]["nlags"][y, x] = int(np.sum(fit))
                tracked[(suffix, x, y)] = position, fit
            catde[:, y, x] = ca_tde(average_ds, x, y, dead_values)
        delays = tde_fields(ds, tde, dead_values, live)

    lags = _common_lags(found, ds, averages)
    positions = {}
    for suffix in TRACKS:
        r, z = np.full((ny, nx, lags.size), np.nan), np.full(
            (ny, nx, lags.size), np.nan
        )
        fitted = np.zeros((ny, nx, lags.size), bool)
        for x, y in found:
            position, fit = tracked[(suffix, x, y)]
            r[y, x], z[y, x], fitted[y, x] = position[:, 0], position[:, 1], fit
        positions.update(
            {f"pos_r_{suffix}": r, f"pos_z_{suffix}": z, f"fit_{suffix}": fitted}
        )

    dims = ("y", "x")
    per_pixel = {}
    for suffix in TRACKS:
        for k in ("vr", "vz", "nlags", "level"):
            per_pixel[f"{k}_{suffix}"] = (dims, out[suffix][k])
    per_pixel["nevents"] = (dims, count)
    per_pixel.update({f"{k}_tde": (dims, v) for k, v in delays.items()})
    per_pixel.update(
        {f"{k}_catde": (dims, catde[i]) for i, k in enumerate(CATDE_VARIABLES)}
    )
    # where() makes the integer counts float64, NaN at dead pixels, as in velocities.nc.
    result = xr.Dataset(per_pixel, coords={"R": ds.R, "Z": ds.Z}).where(
        ~xr.DataArray(dead_values, dims=dims)
    )
    return (
        result.assign(
            **{k: (dims + ("time",), v) for k, v in positions.items()},
            dead=(dims, dead_values),
        )
        .assign_coords(time=lags)
        .assign_attrs(
            min_cc=float(tde.min_cc),
            averages=stamp(averages),
            tracking=stamp(tracking),
            tde=stamp(tde),
            dead_mask_source=source,
        )
    )


# ---------------------------------------------------------------------------------------
# Blob parameters
# ---------------------------------------------------------------------------------------


def pixel_blob(average_ds, x, y, frames, dt, neighbour_step, blob_settings):
    """The 13 blob parameters at reference (x, y), in the units of R, Z and time; NaN
    where an estimate fails.

    The duration time and asymmetry come from the PSD of the pixel's own record
    (``frames``, sampled every ``dt``); the rest from its average: the contour at zero
    lag at the level the average implies (as the centroid track reads it), the FWHM
    along the reference's row and column, and the penalised ellipse fit.
    """
    out = dict.fromkeys(BLOB_PARAMETERS, np.nan)
    out["nevents"] = nevents(average_ds)
    out["taud"], out["lam"] = DurationTimeEstimator(
        SecondOrderStatistic.PSD, Analytics.TwoSided
    ).estimate_duration_time(
        frames,
        dt,
        cutoff=blob_settings.taud_estimation.cutoff,
        nperseg=blob_settings.taud_estimation.nperseg,
    )
    if not out["nevents"]:
        return out
    try:
        out["level"] = neighbour_level(average_ds, "cond_av", neighbour_step)
        contour = get_contour_evolution(average_ds.cond_av, out["level"]).sel(time=0)
        out["area"] = contour.area.item()
        out["lx_c"], out["ly_c"] = contour["size"].values
        out["theta_c"] = contour.theta.item()
    except Exception:
        pass
    try:
        rp, rn, zp, zn = estimate_fwhm_sizes(average_ds)
        out["lr"], out["lz"] = rp - rn, zp - zn
    except Exception:
        pass
    try:
        g = blob_settings.gauss_fit
        # None, the default, is the full extent of the grid; passed on only when set.
        size_max = {} if g.size_max is None else {"size_max": g.size_max}
        out["lx_f"], out["ly_f"], out["theta_f"] = fit_ellipse_to_event(
            average_ds.cond_av,
            x,
            y,
            size_penalty_factor=g.size_penalty,
            aspect_ratio_penalty_factor=g.aspect_penalty,
            theta_penalty_factor=g.tilt_penalty,
            **size_max,
        )
    except Exception:
        pass
    return out


def blobs(ds, average_at, averages, neighbour_step, blob_settings, dead):
    """The blob parameters at every live pixel of the record ``ds``: the
    ``blob_parameters`` product.

    ``average_at`` is as in ``fields``. The Dataset holds ``BLOB_PARAMETERS``, each
    (y, x) float64 and NaN at dead pixels, and ``dead``; attrs: units, averages,
    neighbour_step, blobs (the settings, stamped), dead_mask_source. ``neighbour_step``
    is the one tracking setting the blob parameters read, for the contour level.
    """
    dead_values, source = _dead(dead, ds)
    ny, nx = dead_values.shape
    dt = float(get_dt(ds))
    out = {k: np.full((ny, nx), np.nan) for k in BLOB_PARAMETERS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y in references(dead_values):
            average_ds = average_at(x, y)
            if average_ds is None:
                raise ValueError(f"no average at the live reference ({x}, {y})")
            frames = ds.frames.isel(x=x, y=y).values
            for k, v in pixel_blob(
                average_ds, x, y, frames, dt, neighbour_step, blob_settings
            ).items():
                out[k][y, x] = v
    dims = ("y", "x")
    return xr.Dataset(
        {**{k: (dims, v) for k, v in out.items()}, "dead": (dims, dead_values)},
        coords={"R": ds.R, "Z": ds.Z},
        attrs={
            "units": BLOB_UNITS,
            "averages": stamp(averages),
            "neighbour_step": int(neighbour_step),
            "blobs": stamp(blob_settings),
            "dead_mask_source": source,
        },
    )


# ---------------------------------------------------------------------------------------
# The paper's cut
# ---------------------------------------------------------------------------------------


def reliable(f, ca=None, min_lags=MIN_LAGS, min_events=MIN_EVENTS, track=TRACK):
    """Pixels where every method returned a number and the 2DCA fit rests on enough
    lags: (y, x) bool, of a ``fields`` Dataset ``f``.

    The 2DCA ``track`` (``TRACK``, the centroid, by default) must rest on at least
    ``min_lags`` lags and ``min_events`` events (the paper's ``MIN_LAGS`` and
    ``MIN_EVENTS``, 8 and 200, by default), and its velocity and both TDE off the record
    must be finite; with ``ca``, a Dataset whose ``vr`` (y, x) is another estimate, that
    must be finite too. Edge pixels are out: the 2DCA track leaves the view within a few
    lags there and the TDE loses half its neighbour pairs.
    """
    ok = (f["nlags" + track] >= min_lags) & (f.nevents >= min_events)
    for v in ("vr" + track, "vz" + track, "vr3_tde", "vz3_tde", "vr2_tde", "vz2_tde"):
        ok &= np.isfinite(f[v])
    if ca is not None:
        ok &= np.isfinite(ca.vr)
    ny, nx = ok.shape
    interior = np.zeros((ny, nx), bool)
    interior[1:-1, 1:-1] = True
    return ok & interior

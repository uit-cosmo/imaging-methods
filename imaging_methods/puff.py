"""When the gas puff is on: the part of a raw APD record worth analysing.

Moved from fusion_scripts `density_scan/puff.py` at 12b8543. The drivers that read the
raw files and the discharge database (``shot_window``, ``table_row`` and the command
line) stay there.

``puff_window`` finds the gas puff in the array mean of a raw record and cuts the
discharge window to it; ``find_puff`` does the same on a 1-D array mean. The rule, with
its math and the constants below, is in the README ("The gas-puff window"), and the
stored files carry it in words as ``puff_rule`` (``PuffWindow.rule``).

Use ``puff_window`` on a whole record, never on one already cut to the discharge
window, whose start may be after the puff: the baseline is the start of what it is
given.

    w = puff_window(record, (t_start, t_end))  # record: frames (y, x, time), time in s
    w.found, w.analysis_window, w.note  # what was decided, and why; w.cut is the
                                        # fraction left out
    w.time, w.signal  # the smoothed array mean, at most 4000 points, to draw
    w.baseline, w.threshold, w.level  # its horizontal lines; w.fallback: level is the
                                      # 90th percentile
    w.start, w.end, w.dips  # the puff, NaN when not found, and the dips inside the
                            # window
    w.scatter_over_rise, w.rise_over_excursion  # the record's own values of the two
                                                # tests, at the level used
    w.to_dataset()  # to store it; PuffWindow.from_dataset reads it back
"""

import math
from dataclasses import dataclass, fields

import numpy as np
import xarray as xr

__all__ = [
    "SMOOTH",
    "BASELINE_SPAN",
    "MIN_ON",
    "MIN_DIP",
    "MIN_SPAN",
    "FLAT_START",
    "CLEAR_RISE",
    "FALLBACK_QUANTILE",
    "MAX_POINTS",
    "MAX_DIPS_NAMED",
    "RULE",
    "PuffWindow",
    "array_mean",
    "running_mean",
    "find_puff",
    "puff_window",
]

SMOOTH = 1e-3  # s: running mean of the array mean
BASELINE_SPAN = 5e-3  # s: the start of the record that gives the dark level
# s: how long the signal must stay across the threshold to count as the puff
MIN_ON = 5e-3
MIN_DIP = 1e-3  # s: dips below the threshold shorter than this are not reported
MIN_SPAN = 20e-3  # s: the puff must overlap the discharge window by at least this
# Scatter of the baseline, as a fraction of the rise, above which the record does not
# start dark.
FLAT_START = 0.04
# Fraction of the largest excursion the window's median must reach, on the plasma side.
CLEAR_RISE = 0.25
# The level where the median is not clearly above the dark level: this quantile of the
# window.
FALLBACK_QUANTILE = 0.9
MAX_POINTS = 4000  # samples of the smoothed signal kept for drawing
MAX_DIPS_NAMED = 3

RULE = (
    "Gas puff on where the array-mean light, smoothed over 1 ms, crosses halfway between its dark level at the "
    "start of the record and its median over the discharge window (its 90th percentile where the median is not "
    "clearly above the dark level because most of the window is dark), and stays across for at least 5 ms; the "
    "analysis window is the overlap of the span from the first such crossing to the last point still across, "
    "with the discharge window."
)


@dataclass(frozen=True, eq=False)  # arrays inside: compare fields, not the objects
class PuffWindow:
    """The puff found in one record, with everything a figure of it needs.

    All times are in seconds, signal values in the record's units (volts).
    ``analysis_window`` is ``discharge_window`` whenever ``found`` is False, and
    ``note`` then says why.
    """

    found: bool
    analysis_window: tuple  # (start, end)
    discharge_window: tuple  # (t_start, t_end) as given
    record_window: tuple  # (first, last) sample time of the record
    start: float  # first crossing of the threshold that holds; NaN when not found
    end: float  # last point still across the threshold; NaN when not found
    baseline: float  # dark level at the start of the record
    # The light: the median over the discharge window, or its 90th percentile
    # (fallback_quantile).
    level: float
    # The quantile of the window that ``level`` is: 0.5, or 0.9 (0.1 if light reads
    # lower).
    level_quantile: float
    threshold: float  # halfway between baseline and level
    # Std of the smoothed signal over the start of the record: how flat the dark level
    # is.
    scatter: float
    rise: float  # from the baseline to the level, positive towards the plasma
    # From the baseline to the window's largest excursion (5th/95th percentile), towards
    # the plasma.
    excursion: float
    dips: tuple  # ((start, end), ...) below the threshold inside the analysis window
    note: str
    time: np.ndarray  # the smoothed array mean, for drawing
    signal: np.ndarray

    @property
    def cut(self):
        """Fraction of the discharge window the analysis window leaves out."""
        t0, t1 = self.discharge_window
        a0, a1 = self.analysis_window
        return float(((a0 - t0) + (t1 - a1)) / (t1 - t0)) if t1 > t0 else 0.0

    @property
    def fallback(self):
        """True when ``level`` is the window's 90th percentile (the 10th where light
        reads lower) and not its median, because most of the window is dark and the
        median is the dark level."""
        return self.level_quantile != 0.5

    @property
    def scatter_over_rise(self):
        """How flat the start of the record is, against the rise (``FLAT_START`` is the
        limit)."""
        return float(self.scatter / self.rise) if self.rise > 0 else math.inf

    @property
    def rise_over_excursion(self):
        """How far the window's median is towards its largest excursion
        (``CLEAR_RISE`` is the limit)."""
        return float(self.rise / self.excursion) if self.excursion > 0 else 0.0

    @property
    def rule(self):
        """The rule in words when the window was cut to the puff, or why the discharge
        window was kept; with the note for this record. This is what the preprocessed
        file stores as ``puff_rule``."""
        if self.found:
            return f"{RULE} {self.note[:1].upper()}{self.note[1:]}"
        return f"Discharge window kept: {self.note}"

    def to_dataset(self):
        """The result as a Dataset, to store and to read back with ``from_dataset``:
        ``signal`` over ``time``, ``dips`` (dip, edge), and the rest in attrs."""
        dips = np.asarray(self.dips, dtype=float).reshape(-1, 2)
        return xr.Dataset(
            {"signal": ("time", self.signal), "dips": (("dip", "edge"), dips)},
            coords={"time": self.time},
            attrs={
                f.name: _ATTRS[f.type][0](getattr(self, f.name))
                for f in fields(self)
                if f.name not in _VARIABLES
            },
        )

    @classmethod
    def from_dataset(cls, ds):
        return cls(
            **{
                f.name: _ATTRS[f.type][1](ds.attrs[f.name])
                for f in fields(cls)
                if f.name not in _VARIABLES
            },
            dips=tuple(tuple(map(float, d)) for d in ds["dips"].values.reshape(-1, 2)),
            time=ds["time"].values,
            signal=ds["signal"].values,
        )


# How ``to_dataset`` stores each field of a PuffWindow as an attribute, and how
# ``from_dataset`` reads it back, by its annotation: ``(store, read)``. The arrays are
# variables instead.
_VARIABLES = ("dips", "time", "signal")
_ATTRS = {
    bool: (int, bool),
    tuple: (lambda v: list(map(float, v)), lambda v: tuple(map(float, v))),
    float: (float, float),
    str: (str, str),
}


def array_mean(frames):
    """Mean over the pixels at each sample, float64: ``frames`` is a ``(y, x, time)``
    DataArray (lazily opened is fine, it is read a row at a time). Pixels that are not
    finite over the whole record are left out, as the older raw files hold NaN at their
    dead pixels."""
    frames = frames.transpose("y", "x", "time")
    total = np.zeros(frames.sizes["time"])
    count = 0
    for y in range(frames.sizes["y"]):
        row = np.asarray(frames.isel(y=y).values, dtype=np.float64)  # (x, time)
        ok = np.isfinite(row).all(-1)
        total += row[ok].sum(0)
        count += int(ok.sum())
    if not count:
        raise ValueError("no pixel is finite over the whole record")
    return total / count


def running_mean(x, n):
    """The mean of every ``n`` consecutive samples of ``x`` (length ``x.size - n + 1``)."""
    x = np.asarray(x, dtype=np.float64)
    offset = x[0]
    c = np.concatenate(([0.0], np.cumsum(x - offset)))
    return (c[n:] - c[:-n]) / n + offset


def _runs(mask):
    """``[(first, stop), ...]`` of the maximal runs of True in ``mask``."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _span(a, b):
    return f"{a:.3f}-{b:.3f} s"


def _percentile(q):
    """``0.9`` -> ``"90th percentile"``: the fallback quantile, 0.9 or 0.1."""
    return f"{round(100 * q)}th percentile"


def find_puff(
    time,
    mean,
    discharge_window,
    *,
    smooth=SMOOTH,
    baseline_span=BASELINE_SPAN,
    min_on=MIN_ON,
    min_dip=MIN_DIP,
    min_span=MIN_SPAN,
    flat_start=FLAT_START,
    clear_rise=CLEAR_RISE,
    fallback_quantile=FALLBACK_QUANTILE,
):
    """The puff in an array mean ``mean`` sampled at ``time`` (the whole record), by the
    rule in the README, as a ``PuffWindow``. ``discharge_window`` is
    ``(t_start, t_end)``, and the times are in seconds: the constants and keyword
    arguments with a duration are too. ``fallback_quantile=None`` turns the fallback
    level off, which is the rule with the median alone."""
    t = np.asarray(time, dtype=np.float64)
    x = np.asarray(mean, dtype=np.float64)
    if t.ndim != 1 or t.shape != x.shape or t.size < 2:
        raise ValueError(
            "time and mean must be 1-D arrays of the same length, at least 2 samples"
        )
    if not np.isfinite(x).all():
        raise ValueError("the array mean has non-finite samples")
    t0, t1 = float(discharge_window[0]), float(discharge_window[1])
    if not t1 > t0:
        raise ValueError(f"discharge window {t0}..{t1} s is empty")
    dt = float(np.median(np.diff(t)))
    n = max(1, int(round(smooth / dt)))
    if x.size < n + 2:
        raise ValueError("the record is shorter than the smoothing window")

    sm = running_mean(x, n)
    ts = t[n // 2 : n // 2 + sm.size]
    inside = (ts >= t0) & (ts <= t1)
    if inside.sum() < 2:
        raise ValueError(
            f"discharge window {t0}..{t1} s lies outside the record {t[0]:.4f}..{t[-1]:.4f} s"
        )
    window = sm[inside]
    nb = max(2, int(round(baseline_span / dt)))
    start_of_record = sm[:nb]
    baseline = float(np.median(start_of_record))
    scatter = float(start_of_record.std())
    level = float(np.median(window))
    level_quantile = 0.5

    up = float(np.quantile(window, 0.95)) - baseline
    down = baseline - float(np.quantile(window, 0.05))
    sign = 1.0 if up >= down else -1.0
    excursion = max(up, down)
    rise = sign * (level - baseline)
    above = "above" if sign > 0 else "below"

    # The fallback, only where the median fails the clear-rise test: most of the window
    # is dark after a short puff, and the median is the dark level. The light is then in
    # the window's upper tail (the lower, where light reads lower), so its 90th
    # percentile is the level, provided the clear-rise and flat-start tests pass at that
    # level. A record that does not start dark fails the second, and keeps the median's
    # verdict and its note.
    lead = ""
    if fallback_quantile is not None and (rise <= 0 or rise < clear_rise * excursion):
        q = fallback_quantile if sign > 0 else round(1.0 - fallback_quantile, 12)
        alternative = float(np.quantile(window, q))
        alternative_rise = sign * (alternative - baseline)
        if (
            alternative_rise > 0
            and alternative_rise >= clear_rise * excursion
            and scatter <= flat_start * alternative_rise
        ):
            lead = (
                f"the median over the discharge window ({level:.3f} V) is not clearly {above} the level at the start "
                f"of the record ({baseline:.3f} V): most of the window is dark, so its {_percentile(q)} "
                f"({alternative:.3f} V) is taken as the light level"
            )
            level, level_quantile, rise = alternative, q, alternative_rise
    threshold = 0.5 * (baseline + level)
    light = "median" if level_quantile == 0.5 else _percentile(level_quantile)

    stride = max(1, math.ceil(sm.size / MAX_POINTS))

    def result(found, analysis, note, start=math.nan, end=math.nan, dips=()):
        if lead:
            note = f"{lead}; {note}"
        return PuffWindow(
            found=found,
            analysis_window=tuple(map(float, analysis)),
            discharge_window=(t0, t1),
            record_window=(float(t[0]), float(t[-1])),
            start=float(start),
            end=float(end),
            baseline=baseline,
            level=level,
            level_quantile=level_quantile,
            threshold=threshold,
            scatter=scatter,
            rise=rise,
            excursion=excursion,
            dips=tuple(dips),
            note=note,
            time=ts[::stride].copy(),
            signal=sm[::stride].copy(),
        )

    kept = (t0, t1)
    if rise <= 0:
        return result(
            False,
            kept,
            f"no clear rise: the median over the discharge window ({level:.3f} V) is not {above} the level at the "
            f"start of the record ({baseline:.3f} V): the record starts with the light on, or most of the window "
            f"is dark.",
        )
    if rise < clear_rise * excursion:
        return result(
            False,
            kept,
            f"no clear rise: the median over the discharge window ({level:.3f} V) is only {rise:.3f} V {above} the "
            f"level at the start of the record ({baseline:.3f} V), against {excursion:.3f} V for the window's "
            f"largest excursion: the record starts with the light on, or most of the window is dark.",
        )
    if scatter > flat_start * rise:
        return result(
            False,
            kept,
            f"already on at the record's start: its first {baseline_span * 1e3:g} ms vary by "
            f"{100 * scatter / rise:.0f}% of the rise to the window's {light}, so the record has no dark level "
            f"to compare with.",
        )

    on = sign * (sm - threshold) > 0
    min_run = max(1, int(round(min_on / dt)))
    long_runs = [(a, b) for a, b in _runs(on) if b - a >= min_run]
    if not long_runs:
        return result(
            False,
            kept,
            f"no clear rise: the signal never stays across the threshold for {min_on * 1e3:g} ms.",
        )
    first, last = long_runs[0][0], long_runs[-1][1] - 1
    if first == 0:
        return result(
            False,
            kept,
            "already on at the record's start: its first smoothed sample is across the threshold.",
        )
    start, end = float(ts[first]), float(ts[last])
    # Across the threshold to the end of the record: it does not end there.
    still_on = last == sm.size - 1
    # A cut shorter than the smoothing window is below the resolution of the threshold
    # crossing: not made.
    a0 = start if start - t0 >= smooth else t0
    a1 = end if not still_on and t1 - end >= smooth else t1
    if a1 - a0 < min_span:
        return result(
            False,
            kept,
            f"the puff ({_span(start, end)}) overlaps the discharge window by only {max(a1 - a0, 0) * 1e3:.0f} ms.",
            start,
            end,
        )

    dips = []
    for i, j in _runs(~on[first : last + 1]):
        d0, d1 = max(float(ts[first + i]), a0), min(float(ts[first + j - 1]), a1)
        if d1 - d0 >= min_dip:
            dips.append((d0, d1))

    if a0 > t0:
        parts = [
            f"puff rises at {start:.3f} s, so the first {(a0 - t0) * 1e3:.0f} ms of the window are dark"
        ]
    else:
        parts = [
            f"puff already on at the window's start (on from {start:.3f} s), nothing cut there"
        ]
    if a1 < t1:
        parts.append(
            f"light gone at {end:.3f} s, so the last {(t1 - a1) * 1e3:.0f} ms are cut"
        )
    if dips:
        named = ", ".join(_span(*d) for d in dips[:MAX_DIPS_NAMED])
        more = (
            f" and {len(dips) - MAX_DIPS_NAMED} more"
            if len(dips) > MAX_DIPS_NAMED
            else ""
        )
        parts.append(
            f"dip{'s' if len(dips) > 1 else ''} below threshold at {named}{more}"
        )
    return result(True, (a0, a1), "; ".join(parts) + ".", start, end, dips)


def puff_window(ds, discharge_window, **kwargs):
    """``find_puff`` on a record: ``ds`` holds ``frames`` (y, x, time) with its
    ``time`` coordinate, or is that DataArray, and ``discharge_window`` is
    ``(t_start, t_end)``. The array mean over the pixels is the signal. Keyword
    arguments are ``find_puff``'s."""
    frames = ds["frames"] if isinstance(ds, xr.Dataset) else ds
    return find_puff(
        frames["time"].values, array_mean(frames), discharge_window, **kwargs
    )

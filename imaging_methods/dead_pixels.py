"""Dead APD pixels, estimated from the raw record.

Moved from fusion_scripts `density_scan/dead_pixels.py` at 12b8543. The drivers that
read the raw files and the discharge database (``estimate_shot``, ``estimate_shots``,
``discharge_window``, ``process_pool`` and the command line) stay there.

A pixel is live when it sees the plasma, and the evidence for that is in its
fluctuations: their spectrum, and whether they follow a live neighbour's. Its PDF alone
cannot tell. A dim pixel inside the separatrix has a narrow, near-Gaussian PDF just as
dead ones do, and a dead channel can be skewed by rare spikes or by telegraph noise.
Everything here is computed on the raw frames over the window ``estimate`` is given.
fusion_scripts' ``estimate_shot`` gives it the shot's analysis window: the discharge
window cut to the gas puff (``imaging_methods.puff``), so that the dark stretch before
the puff does not dilute the spectra.

1. Red spectrum. The Welch PSD of every pixel is reduced to the median over the blob
   band, 1-20 kHz, divided by the median over 300-900 kHz, where only digitizer noise
   is left. The medians ignore the narrow pickup lines in that upper band. Noise is
   white (a ratio near 1) and plasma fluctuations are red, so a pixel whose ratio
   exceeds ``RED`` is live.
2. Neighbours. A pixel below that is still live if its 2-20 kHz signal follows an
   already-live pixel among its eight neighbours, i.e. the regression gain of its
   band-passed signal on the neighbour's, ``corr * std_pixel / std_neighbour``, exceeds
   ``GAIN``. Repeated until no pixel joins. This keeps dim pixels whose own
   fluctuations are weak, which lie mostly in the innermost columns. Crosstalk, a dead
   channel carrying a few percent of a neighbour's signal, stays below the gain. The
   gain on a live neighbour is used rather than the correlation, because dead channels
   share pickup with one another and correlate strongly while carrying nothing.
3. Every pixel not reached is dead. A pixel with no samples (NaN or constant) is dead.
4. Run day. Hardware does not change within a run day: ``consolidate`` marks a pixel
   dead in every shot of the day when it is dead in at least ``DAY_FRACTION`` of them.
   This fixes the one failure of the per-shot rule seen so far. On the later 1120814
   shots, two dead channels, (6, 3) and (9, 3), share telegraph noise with a dim live
   neighbour and pass step 2.

Checked on all 111 raw Alcator C-Mod APD shots on the group server (2026-10-07):

- It reproduces the hand-made 1160616 mask (``experimental_database``'s
  ``hand_made_dead_pixel_mask``) on all nine shots and on every other 2016 day.
- 2012-02 to 2015-09 give the same 18 dead pixels on every day: the 2016 mask without
  (4, 8), (8, 7), (8, 8) and (9, 7), which died between 1150916 and 1160616.
- The result is the same for any ``RED`` from 100 to 300; dead channels reach at most
  ~100.
- No reference exists for 2009-2011: those days give 6, 22 (one shot, 2010) and 8 dead
  pixels.
- Over the puff window (2026-10-08) the verdict of every one of the 111 shots, and so
  the mask of every run day, is the same as over the discharge window. 38 shots'
  windows are cut; the number of pixels that live only through a neighbour changes by
  1 to 5 in five of them.
"""

import numpy as np
import xarray as xr
from scipy import ndimage, signal

__all__ = [
    "RED_BAND",
    "NOISE_BAND",
    "BLOB_BAND",
    "RED",
    "GAIN",
    "DAY_FRACTION",
    "NPERSEG",
    "OFFSETS",
    "NO_DATA",
    "NO_EVIDENCE",
    "FOLLOWS_NEIGHBOUR",
    "RED_SPECTRUM",
    "red_ratio",
    "band_passed",
    "neighbour_gains",
    "estimate",
    "consolidate",
]

RED_BAND = (1e3, 20e3)  # Hz: where the plasma's fluctuation power is
NOISE_BAND = (300e3, 900e3)  # Hz: digitizer noise and pickup lines only
BLOB_BAND = (2e3, 20e3)  # Hz: band-pass of the neighbour regression
RED = 150.0  # PSD ratio above which a pixel sees the plasma by itself
GAIN = 0.1  # regression gain on a live neighbour above which a pixel follows it
DAY_FRACTION = (
    1 / 3
)  # dead in at least this fraction of a run day's shots -> dead all day
NPERSEG = 4096
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
# Per-pixel evidence codes, stored with the mask so a reader can see why each pixel came
# out as it did.
NO_DATA, NO_EVIDENCE, FOLLOWS_NEIGHBOUR, RED_SPECTRUM = -1, 0, 1, 2


def red_ratio(frames, dt):
    """``(ratio, frequency, psd)``: the median PSD over RED_BAND over the median over
    NOISE_BAND, per pixel."""
    ny, nx, _ = frames.shape
    f = None
    psd = None
    for y in range(ny):
        for x in range(nx):
            s = frames[y, x].astype(np.float64)
            f, p = signal.welch(s - s.mean(), fs=1 / dt, nperseg=NPERSEG)
            if psd is None:
                psd = np.zeros((ny, nx, f.size))
            psd[y, x] = p
    low = np.median(psd[..., (f >= RED_BAND[0]) & (f <= RED_BAND[1])], axis=-1)
    high = np.median(psd[..., (f >= NOISE_BAND[0]) & (f <= NOISE_BAND[1])], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return low / high, f, psd


def band_passed(frames, dt, band=BLOB_BAND):
    """Block averages over 1/(2 f_high), less their running mean over 1/f_low; one row
    at a time, to keep a float64 copy of the record out of memory."""
    block = max(1, int(round(1 / (2 * band[1]) / dt)))
    n = frames.shape[-1] // block
    width = max(3, int(round(1 / band[0] / (dt * block))))
    out = np.empty((*frames.shape[:-1], n))
    for y in range(frames.shape[0]):
        lp = (
            frames[y, :, : n * block]
            .astype(np.float64)
            .reshape(frames.shape[1], n, block)
            .mean(-1)
        )
        out[y] = lp - ndimage.uniform_filter1d(lp, width, axis=-1, mode="nearest")
    return out


def neighbour_gains(bp):
    """``(ny, nx, 8)``: regression gain of each pixel's band-passed signal on each
    neighbour's, NaN off the array."""
    ny, nx, _ = bp.shape
    std = bp.std(-1)
    z = bp / np.where(std > 0, std, 1)[..., None]
    gains = np.full((ny, nx, len(OFFSETS)), np.nan)
    for y in range(ny):
        for x in range(nx):
            for k, (dy, dx) in enumerate(OFFSETS):
                q, p = y + dy, x + dx
                if 0 <= q < ny and 0 <= p < nx and std[q, p] > 0:
                    gains[y, x, k] = np.mean(z[y, x] * z[q, p]) * std[y, x] / std[q, p]
    return gains


def estimate(ds, red=RED, gain=GAIN, keep_psd=False):
    """The dead-pixel mask of one record, with the evidence behind it.

    ``ds`` holds ``frames`` (y, x, time) and its ``time`` coordinate, already restricted
    to the window to judge. Returns a Dataset with ``dead`` (y, x), ``evidence`` (the
    codes above), ``red_ratio`` and ``gain`` (the best gain on a live neighbour), plus
    ``psd`` (y, x, frequency) when ``keep_psd``.
    """
    frames = np.array(ds.frames.transpose("y", "x", "time").values)
    dt = float(ds.time[1] - ds.time[0])
    valid = np.isfinite(frames).all(-1)
    valid[valid] = frames[valid].std(-1) > 0
    frames[~valid] = (
        0  # no samples: neither red nor followed, so dead with NO_DATA below
    )
    ratio, freq, psd = red_ratio(frames, dt)
    ratio = np.where(valid, ratio, np.nan)
    gains = neighbour_gains(band_passed(frames, dt))

    ny, nx = valid.shape
    evidence = np.full((ny, nx), NO_EVIDENCE, dtype=np.int8)
    evidence[~valid] = NO_DATA
    evidence[valid & (ratio > red)] = RED_SPECTRUM
    best = np.full((ny, nx), np.nan)
    changed = True
    while changed:
        changed = False
        live = evidence > NO_EVIDENCE
        for y, x in np.argwhere(evidence == NO_EVIDENCE):
            on_live = [
                gains[y, x, k]
                for k, (dy, dx) in enumerate(OFFSETS)
                if 0 <= y + dy < ny and 0 <= x + dx < nx and live[y + dy, x + dx]
            ]
            if on_live:
                best[y, x] = np.nanmax(on_live)
                if best[y, x] > gain:
                    evidence[y, x] = FOLLOWS_NEIGHBOUR
                    changed = True

    out = xr.Dataset(
        {
            "dead": (("y", "x"), evidence <= NO_EVIDENCE),
            "evidence": (("y", "x"), evidence),
            "red_ratio": (("y", "x"), ratio),
            "gain": (("y", "x"), best),
        },
        coords={k: ds[k] for k in ("R", "Z") if k in ds.coords},
        attrs={
            # The stored files carry this text: kept as it was when the code moved.
            "method": "density_scan.dead_pixels: red spectrum, or following a live neighbour",
            "red_band_hz": list(RED_BAND),
            "noise_band_hz": list(NOISE_BAND),
            "blob_band_hz": list(BLOB_BAND),
            "red_threshold": float(red),
            "gain_threshold": float(gain),
            "evidence_codes": "-1 no data, 0 dead, 1 follows a live neighbour, 2 red spectrum",
        },
    )
    if keep_psd:
        out["psd"] = (("y", "x", "frequency"), psd.astype(np.float32))
        out = out.assign_coords(frequency=freq)
    return out


def consolidate(masks, fraction=DAY_FRACTION):
    """One run day's mask from its shots' masks: dead where dead in at least
    ``fraction`` of them."""
    stacked = np.stack([np.asarray(m, dtype=bool) for m in masks])
    return stacked.mean(0) >= fraction - 1e-12

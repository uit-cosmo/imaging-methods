"""The synthetic record tests/test_decorrelation.py runs on, the settings it runs at, and
how its outputs are compared with the expected ones in tests/data/.

The record and the expected values were made by fusion_scripts'
`decorrelation/tests/make_pipeline_fixture.py` at 7b761fd, before the decorrelation
pipeline existed, so the expected values are the old implementation's own:

- ``pipeline_expected_fields.nc``: every method's velocity field, as the paper's
  ``apd<shot>_velocities.nc`` holds it, with the contour level of the centroid track
  (``level_com``) and the tracked positions and fitted lags (``pos_r_*``, ``pos_z_*``,
  ``fit_*``);
- ``pipeline_expected_blobs.nc``: the 13 blob parameters.

They cannot be remade here, since the code that made them is not in this package.
``make_record`` is how the record was made, and a test holds it to the stored file.

The record is C-Mod-like but small, so that the tests stay quick: a 5 x 4 view at the
1160616027 pitch, R and Z in metres, sampled every 3 us, of Gaussian pulses moving at
(500, 150) m/s, normalized per pixel. Two pixels are dead and filled from their
neighbours, as the preprocessed files fill them, and one live pixel is uniform noise that
never reaches the 2DCA threshold, so its average is the empty one. Frames are stored as
float32 and widened to float64 on load, which is exact.

The settings (``settings``) are the C-Mod defaults with one knob moved in every group:
the 2DCA threshold and window, both mask signal factors, an ellipse penalty and the
Welch segment. A setting the pipeline failed to pass on would then show up as a
mismatch, where the defaults would hide it.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from imaging_methods import (
    GaussFitParams,
    PositionFilterParams,
    TaudEstimationParams,
    VelocityParams,
    decorrelation,
)

DATA = Path(__file__).parent / "data"
RECORD = DATA / "pipeline_record.nc"
EXPECTED_FIELDS = DATA / "pipeline_expected_fields.nc"
EXPECTED_BLOBS = DATA / "pipeline_expected_blobs.nc"

NY, NX, NT = 5, 4, 4000
DT = 3e-6  # [s]
DR, DZ = 3.77e-3, 3.86e-3  # [m]
R0, Z0 = 0.8803, -0.0428
VR, VZ = 500.0, 150.0  # [m/s]
ELL, TAU = 3.5e-3, 12e-6  # pulse width [m] and lifetime [s]
PULSES = 6000
NOISE = 0.05
SEED = 20261007
DEAD = ((0, 1), (1, 2))  # (x, y): one on the edge, one inside
FLAT = (3, 0)  # (x, y): live, never above the threshold
CROSS_CORR_MASK_SIGNAL_FACTOR = 0.7


def make_record():
    """The record, from a fixed legacy-stream seed."""
    rng = np.random.RandomState(SEED)
    r = R0 + DR * np.arange(NX)
    z = Z0 + DZ * np.arange(NY)
    R, Z = np.meshgrid(r, z)  # (y, x)
    t = DT * np.arange(NT)
    reach = 5 * TAU
    pad = 3 * ELL + reach * max(abs(VR), abs(VZ))
    t0 = rng.uniform(-reach, t[-1] + reach, PULSES)
    r0 = rng.uniform(r[0] - pad, r[-1] + pad, PULSES)
    z0 = rng.uniform(z[0] - pad, z[-1] + pad, PULSES)
    amplitude = rng.exponential(1.0, PULSES)
    half = int(round(reach / DT))
    frames = np.zeros((NY, NX, NT))
    for tk, rk, zk, ak in zip(t0, r0, z0, amplitude):
        i = int(round(tk / DT))
        lo, hi = max(i - half, 0), min(i + half + 1, NT)
        if lo >= hi:
            continue
        s = t[lo:hi] - tk
        dr = R[..., None] - rk - VR * s
        dz = Z[..., None] - zk - VZ * s
        frames[:, :, lo:hi] += ak * np.exp(-(dr**2 + dz**2) / ELL**2 - (s / TAU) ** 2)
    frames += NOISE * rng.standard_normal(frames.shape)
    frames = (frames - frames.mean(-1, keepdims=True)) / frames.std(-1, keepdims=True)

    x, y = FLAT
    flat = rng.uniform(-1.0, 1.0, NT)
    frames[y, x] = (flat - flat.mean()) / flat.std()
    for x, y in DEAD:
        # As the preprocessed files fill a dead pixel: the mean of the two either side
        # along R, or a copy of the one there is at the edge.
        neighbours = [
            frames[y, q] for q in (x - 1, x + 1) if 0 <= q < NX and (q, y) not in DEAD
        ]
        frames[y, x] = np.mean(neighbours, axis=0)

    dead = np.zeros((NY, NX), bool)
    for x, y in DEAD:
        dead[y, x] = True
    return xr.Dataset(
        {
            "frames": (("y", "x", "time"), frames.astype(np.float32)),
            "dead": (("y", "x"), dead),
        },
        coords={
            "R": (("y", "x"), R.astype(np.float32)),
            "Z": (("y", "x"), Z.astype(np.float32)),
            "time": t,
        },
        attrs={
            "shot_number": 0,
            "dead_mask_source": "synthetic, pixels (0, 1) and (1, 2)",
            "description": "decorrelation/tests/make_pipeline_fixture.py",
        },
    )


def load_record(path=RECORD):
    """The stored record as the pipeline takes it: in memory, frames float64, R and Z in
    metres."""
    ds = xr.load_dataset(path)
    return ds.assign(frames=ds.frames.astype(np.float64))


def settings():
    """``(averages, tracking, tde, blobs)``: the C-Mod defaults with one knob moved in
    every group, spelled out."""
    averages = decorrelation.Averages(
        threshold=2.0,
        window=10,  # even: the 2DCA widens it to 11 lags
        check_max=1,
        single_counting=True,
    )
    tracking = decorrelation.Tracking(
        position_filter=PositionFilterParams(
            window_size=1,
            window_type="hann",
            mask_distance=2,
            mask_signal_factor=0.6,
            require_within_boundaries=False,
        ),
        cross_corr_mask_signal_factor=CROSS_CORR_MASK_SIGNAL_FACTOR,
        velocity=VelocityParams(estimator="lsq"),
        neighbour_step=1,
    )
    blobs = decorrelation.Blobs(
        gauss_fit=GaussFitParams(
            size_penalty=5, aspect_penalty=0.3, tilt_penalty=0.2, size_max=None
        ),
        taud_estimation=TaudEstimationParams(cutoff=1e6, nperseg=1000.0),
    )
    return averages, tracking, decorrelation.Tde(), blobs


# ---------------------------------------------------------------------------------------
# Comparing, as fusion_scripts' `decorrelation/apd_check/regress_pipeline.py` compares
# ---------------------------------------------------------------------------------------

RTOL = 1e-12
SHOWN = 5  # differing points shown per variable


@dataclass
class Comparison:
    """What ``compare`` found: the variables compared, how many are bit-equal, the
    largest relative difference over all of them, and one line per mismatch."""

    names: list
    bit_equal: int = 0
    max_rel: float = 0.0
    failures: list = field(default_factory=list)

    @property
    def passed(self):
        return not self.failures


def _shown(flat, dims, shape, a, b, fmt):
    """The first few of the flat indices ``flat``, each as (dim=index, ...) new vs
    reference."""
    lines = []
    for i in flat[:SHOWN]:
        index = np.unravel_index(i, shape)
        where = ", ".join(f"{d}={int(j)}" for d, j in zip(dims, index))
        lines.append(f"({where}) {a.flat[i]:{fmt}} vs {b.flat[i]:{fmt}}")
    return "; ".join(lines)


def compare(new, ref, names, rtol=RTOL):
    """``new`` against ``ref``, variable by variable over ``names``: the same dims, dtype
    and NaN pattern, and values within ``rtol`` of the reference's, relative (bools and
    integers exactly)."""
    out = Comparison(list(names))
    for k in names:
        if k not in new:
            out.failures.append(f"{k}: missing")
            continue
        a, b = new[k], ref[k]
        if a.dims != b.dims or a.shape != b.shape:
            out.failures.append(f"{k}: dims {dict(a.sizes)}, reference {dict(b.sizes)}")
            continue
        if a.dtype != b.dtype:
            out.failures.append(f"{k}: dtype {a.dtype}, reference {b.dtype}")
            continue
        av, bv = np.asarray(a.values), np.asarray(b.values)
        if av.dtype.kind != "f":
            bad = np.flatnonzero(av != bv)
            if len(bad):
                shown = _shown(bad, a.dims, av.shape, av, bv, "")
                out.failures.append(f"{k}: differs at {len(bad)} of {av.size}: {shown}")
            else:
                out.bit_equal += 1
            continue
        nan = np.isnan(bv)
        bad = np.flatnonzero(np.isnan(av) != nan)
        if len(bad):
            shown = _shown(bad, a.dims, av.shape, av, bv, ".6g")
            out.failures.append(
                f"{k}: NaN pattern differs at {len(bad)} of {av.size}: {shown}"
            )
            continue
        diff = np.where(nan, 0.0, np.abs(av - bv))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(diff == 0, 0.0, diff / np.abs(bv))
        out.max_rel = max(out.max_rel, float(rel.max(initial=0.0)))
        if not diff.any():
            out.bit_equal += 1
        bad = np.flatnonzero(rel > rtol)
        if len(bad):
            bad = bad[np.argsort(-rel.flat[bad], kind="stable")]
            out.failures.append(
                f"{k}: values differ at {len(bad)} of {int((~nan).sum())} finite, max "
                f"rel {float(rel.max()):.3g}: "
                + _shown(bad, a.dims, av.shape, av, bv, ".17g")
            )
    return out


def assert_same(new, ref, names):
    """The regression's test: same dims, dtype and NaN pattern, values within 1e-12
    relative."""
    result = compare(new, ref, names)
    assert result.passed, "\n".join(result.failures)

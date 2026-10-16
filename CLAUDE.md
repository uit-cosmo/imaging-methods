# CLAUDE.md

## What this repo is

`imaging_methods` is a small scientific Python package (uit-cosmo, ~2000 lines)
for analyzing imaging data at the tokamak boundary (Alcator C-Mod APD/GPI,
Phantom camera, and synthetic blobmodel data). The central method is
two-dimensional conditional averaging (2DCA); the other methods (contouring,
ellipse fitting, FWHM sizes, duration-time estimation, velocity estimates)
operate on 2DCA output. Backs a manuscript in preparation for PoP:
"Conditionally averaged blob structures in Alcator C-Mod". Method descriptions
with math live in `README.md`.

Naming gotcha: the directory is `imaging-methods` (hyphen), the package is
`imaging_methods` (underscore), and the git remote is still
`uit-cosmo/phantom` (the repo's old name — GitHub redirects to
`uit-cosmo/imaging-methods`).

## Data format

Everything operates on the cmod_functions-style xarray Dataset
(github.com/sajidah-ahmed/cmod_functions):

- `frames(y, x, time)` — signal; **time is the last dim** in raw data.
- `R(y, x)`, `Z(y, x)` — 2D physical coordinate grids (radial R ↔ x index,
  vertical/poloidal Z ↔ y index). Grids may be **non-uniform** (APD quadrant
  gaps): `get_dr`/`get_dz` return the *mean* spacing, and
  `indexes_to_coordinates` bilinearly interpolates R/Z rather than assuming a
  linear map.
- Reference pixels (`refx`, `refy`) are integer *indices* into x/y, not
  physical coordinates.

## Architecture

All source in `imaging_methods/` (flat); public API is what `__init__.py`
re-exports. The downstream consumer is `../fusion_scripts` — grep it before
changing signatures.

Pipeline order roughly matches module order:

- `method_parameters.py` — nested dataclasses (`MethodParameters` holding
  `PreprocessingParams`, `TwoDcaParams`, `GaussFitParams`, `ContouringParams`,
  `TaudEstimationParams`, `PositionFilterParams`). Every analysis function
  takes one of these instead of loose kwargs. Defaults for synthetic/APD data
  come from `get_default_*_method_params()` in `utils.py`.
- `data_preprocessing.py` — `interpolate_nans_3d` (dead-pixel interpolation;
  NaN mask taken from t=0 and assumed constant in time), `run_norm_ds`
  (running normalization via fppanalysis; zero-fills non-finite output).
- `dead_pixels.py` — dead APD pixels from the raw record: Welch red-spectrum
  ratio, then regression gain on an already-live neighbour, iterated;
  `consolidate` gives a run day's mask. Its loops (per pixel, per row) are
  there for memory: batched, `estimate` peaks at 6 times the record, not 2.5.
- `puff.py` — the gas-puff analysis window of a raw record (`puff_window`/
  `find_puff` → `PuffWindow`, with its rule in words). Give it the whole
  record: the dark level is read off its start. The drivers that read files
  stay in fusion_scripts. Both rules, with their math, are in the README.
- `cond_av.py` — `find_events_and_2dca(ds, TwoDcaParams)`: the core method.
  Returns `(events, average_ds)` where `average_ds` has `cond_av`,
  `cond_repr` (conditional reproducibility = cond_av²/mean²), `cross_corr`,
  `refx/refy/number_events`, on a relative `time` coordinate centered at the
  peak.
- `contours.py` — `get_contour_evolution(event, threshold_factor)`: contour at
  a fraction of max amplitude per frame → Dataset with contours, centroid,
  area, perimeter, convexity deficiency, plus per-frame ellipse fits
  (`skimage.EllipseModel`). Multiple contours → the one enclosing most mass.
- `parameter_estimation.py` — `fit_ellipse`/`fit_ellipse_to_event` (tilted 2D
  Gaussian fit at time lag 0, Nelder-Mead multistart, with size/aspect/tilt
  penalty terms — see README for the error function) and `estimate_fwhm_sizes`
  (interpolated FWHM along the row/column through the reference pixel).
- `fitting.py` — `fit_two_sided` (two-sided exponential centred at 0,
  L-BFGS-B) and `fit_exp` (straight line through log y), for cuts through an
  average.
- `geometry.py` — `calculate_splinted_LCFS`: the outboard leg of EFIT's
  boundary as a cubic spline R(Z); C-Mod's numbers, in cm, hardcoded.
- `maximum_trajectory.py` — sub-pixel peak tracking per frame (2D parabolic
  interpolation, shift clipped to ±0.5 pixel).
- `velocity_estimates.py` — glue: position from contour centroid or max
  trajectory → `smooth_da` (Hann filter) → velocity over a validity mask
  (`get_combined_mask`: near reference AND signal high, longest contiguous
  block). `VelocityParams.estimator` picks how the masked track is reduced:
  `"central_diff"` (average of centered differences — telescopes, so it is
  endpoint-dominated) or `"lsq"` (least squares slope, uses every lag).
  `estimate_velocities_synthetic_ds` runs the whole battery incl. 3TDE via
  `velocity_estimation`.
- `duration_time_estimation.py` — `DurationTimeEstimator`: fits the analytic
  FPP ACF or PSD (expressions in `utils.py`) with `differential_evolution`
  (seeded). For PSD fits `nperseg` is required and critical — a good starting
  point is the run-normalization window length.
- `synthetic_data.py` — bridge to blobmodel 2.0: `get_blob` (imaging-methods
  naming for `Blob`) and `make_2d_realization` (`Model.from_blobs` +
  `layout="imaging"`, yielding the frames/R/Z format above).
- `utils.py` — `PhantomDataInterface`/`MaskedImagingDataInterface`
  (`velocity_estimation` interfaces), analytic `autocorrelation`/
  `power_spectral_density`, `smooth_da`, default method parameters.
- `decorrelation.py` — the 2DCA at every live reference pixel of a record and
  what is read off it (moved from fusion_scripts' `decorrelation/pipeline.py`):
  `average` → `stack` into one bank (`at` gives an average back, bit for bit)
  → `fields` (three 2DCA tracks, 2- and 3-point TDE off the record and on the
  average) and `blobs` (13 blob parameters); `dead_mask(ds, hand_made)`,
  `reliable` (the paper's cut). Its helpers are `contours.neighbour_level`,
  `maximum_trajectory.get_maximum_time` and
  `velocity_estimates.get_positions_and_mask`. Shot Explorer's decorrelation
  products are thin adapters over it.

`examples/synthetic_data.py` is a runnable end-to-end walkthrough on
blobmodel data.

## Commands

```bash
pip install -e .[dev]        # dev extras: pytest, black, mypy, cosmoplots
pytest                       # tests/ at repo root; fast, deterministic
black .                      # code is black-formatted (CI does not check it)
```

There is a project venv at `.venv/`. CI is `.github/workflows/tests.yml`:
`pip install -e .[dev]`, `pip check`, `pytest -v` on Python 3.10/3.11. No
black/mypy gates in CI and no tool config in pyproject.toml — defaults apply.
`commands.txt` holds a gifsicle one-liner for compressing animation gifs
(figures land in `figures/`, some are embedded in README from GitHub URLs).

## Conventions and gotchas

- **Ellipse convention**: `lx < ly` (ly is the larger axis) and
  `theta ∈ [0, π)`, measured from the R axis; swapping the axes rotates theta
  by π/2. Enforced identically in `fit_ellipse` and
  `fit_ellipses_to_contour_ds` — keep them in sync.
- **`lam` (asymmetry)**: rise fraction, `tr = lam * taud`, `lam ∈ [0, 1]` —
  the FPP-literature convention.
- **Dead pixels**: `find_events_and_2dca` assumes every pixel has non-zero
  std; zero-filled/dead pixels produce 0/0 → NaN columns in `cross_corr` and
  `cond_repr`. Interpolate (`interpolate_nans_3d`) or mask first.
- **blobmodel version**: pinned `blobmodel~=2.0.0` (developed locally in
  `../blobmodel`, see its CLAUDE.md). `synthetic_data.py` is now a thin
  wrapper over the 2.0 API: `get_blob` maps imaging-methods naming
  (vx/lx/posx/taup) onto `Blob`, and `make_2d_realization` builds
  `Model.from_blobs(blobs, geometry=Geometry(...))` and returns
  `make_realization(layout="imaging")` directly (no manual frames/R/Z
  conversion). `DeterministicBlobFactory` was removed — blobmodel's
  `BlobListFactory` is the replacement. Note blobmodel 2.0 uses the same
  FPP `lam` convention as this repo (rise fraction).
- **Randomness**: `make_2d_realization`'s default blob getter still draws
  amplitudes/positions from global `np.random` (tests rely on
  `np.random.seed` for determinism); blobmodel 2.0's rng seeding is not
  threaded through it.
- `smooth_da` output carries a boolean `valid` time coordinate — False where
  the input was NaN (values there are extrapolated and can lie far outside
  the field of view).
- **netCDF4 before h5py**: `__init__.py` imports `netCDF4` first.
  `velocity_estimation` pulls in h5py (through plasmapy), and once h5py's
  libhdf5 is loaded every netCDF read fails with `Errno -101`. Keep it the
  package's first import.
- **The decorrelation defaults are cache keys**: the defaults of
  `decorrelation.Averages`, `Tracking`, `Tde` and `Blobs` are the C-Mod
  settings of the paper's twodca deck, written as literals (the APD defaults
  plus the deck's changes). Shot Explorer hashes them field by field, and
  `stamp` writes them with repr, so no field's name, order, annotation, value
  or Python type (`size_penalty` stays the int 5) may change without a new
  cache key downstream. `tests/test_decorrelation.py` pins them.
- `find_events_and_2dca` and `get_contour_evolution` print progress with bare
  `print` (`verbose=True` default); keep output quiet in library additions or
  route through the existing `verbose` flags.
- Docstrings are numpydoc-ish but drift; when a docstring and the code
  disagree about dim order (`(time, x, y)` vs `(y, x, time)`), trust the code
  — downstream code mostly uses order-agnostic xarray selection.

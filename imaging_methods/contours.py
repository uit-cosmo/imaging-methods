import xarray as xr
import numpy as np
from skimage import measure
from shapely import contains_xy
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from scipy.ndimage import map_coordinates
from skimage.measure import EllipseModel

__all__ = [
    "compute_contour_mass",
    "indexes_to_coordinates",
    "is_within_boundaries",
    "fit_ellipses_to_contour_ds",
    "get_contour_evolution",
    "zero_lag_frame",
    "neighbour_ratios",
    "neighbour_level",
]


def is_within_boundaries(contour, shape):
    """Whether a contour closes strictly inside the array border.

    ``contour`` is in index space, (row, col), as returned by
    ``skimage.measure.find_contours``; ``shape`` is the frame's ``(rows, cols)``.
    A contour that reaches the border has been cut off by the field of view, so
    its centroid is held by the array edge rather than by the structure: it stops
    moving while the structure keeps going, which biases any velocity fitted
    through it. An empty contour counts as outside.
    """
    if len(contour) == 0:
        return False
    rows, cols = contour[:, 0], contour[:, 1]
    return bool(
        np.all(rows > 0)
        and np.all(rows < shape[0] - 1)
        and np.all(cols > 0)
        and np.all(cols < shape[1] - 1)
    )


def compute_contour_mass(contour, frame, R, Z):
    """
    Compute the mass (sum of enclosed intensity) of a contour using a Shapely Polygon.

    Parameters
    ----------
    contour : ndarray
        Contour coordinates in (row, col) format, shape (N, 2), from skimage.measure.find_contours.
    frame : ndarray
        2D array of intensity values, shape (rows, cols).
    R : ndarray
        2D array of radial coordinates, shape (rows, cols).
    Z : ndarray
        2D array of axial coordinates, shape (rows, cols).

    Returns
    -------
    float
        Sum of intensity values within the contour, weighted by grid cell area.
        Returns 0.0 for invalid or empty contours.
    """
    if len(contour) < 3:  # Need at least 3 points to form a polygon
        return 0.0

    # Convert contour indices to (r, z) coordinates
    try:
        r_values, z_values = indexes_to_coordinates(R, Z, contour)
        contour_coords = np.stack((r_values, z_values), axis=-1)
    except (ValueError, IndexError):
        return 0.0

    # Create Polygon from contour
    try:
        polygon = Polygon(contour_coords)
        if not polygon.is_valid:
            return 0.0
    except ValueError:
        return 0.0

    # Check which grid points are inside the polygon (vectorized)
    mask_2d = contains_xy(polygon, R, Z)

    # Sum intensity values within the mask
    mass = np.sum(frame[mask_2d])

    return float(mass) if not np.isnan(mass) else 0.0


def indexes_to_coordinates(R, Z, indexes):
    """Convert fractional (row, col) contour indices to (r, z) coordinates by bilinear
    interpolation of the R, Z grids. Supports non-uniform grids (e.g. detectors with
    gaps between quadrants); on a uniform grid this reduces to the linear map
    min + index * spacing. NaN index rows yield NaN coordinates."""
    rows, cols = indexes[:, 0], indexes[:, 1]
    valid = np.isfinite(rows) & np.isfinite(cols)
    r_values = np.full(rows.shape, np.nan)
    z_values = np.full(rows.shape, np.nan)
    # map_coordinates returns the input dtype: integer grids would truncate
    # the interpolated coordinates, so cast to float first.
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    r_values[valid] = map_coordinates(
        R, [rows[valid], cols[valid]], order=1, mode="nearest"
    )
    z_values[valid] = map_coordinates(
        Z, [rows[valid], cols[valid]], order=1, mode="nearest"
    )
    return r_values, z_values


def fit_ellipses_to_contour_ds(contours):
    """
    Fit ellipses to contours in a contour dataset.

    Parameters
    ----------
    contours : xr.Dataset
        Dataset containing contours with dimensions (time, point_idx, coord)

    Returns
    -------
    xr.Dataset
        Dataset with ellipse parameters for each time point
    """
    pos_values = []  # x-center
    l_values = []  # x-length
    theta_values = []  # rotation angle

    # Fit ellipse for each time point
    for t in range(len(contours)):
        # Get contour for this time point, removing NaN rows
        contour = contours[t]
        valid_points = ~np.isnan(contour).all(axis=1)
        contour = contour[valid_points]

        # Skip if not enough points
        if len(contour) < 5:
            pos_values.append([np.nan, np.nan])
            l_values.append([np.nan, np.nan])
            theta_values.append(np.nan)
            continue

        # Fit ellipse
        model = EllipseModel()
        success = model.estimate(contour)

        if success:
            # Unpack parameters: x0, y0, a, b, theta
            rx, ry, lx, ly, theta = model.params
            # Swapping the semi-axes requires rotating theta by pi/2.
            # Same convention as fit_ellipse in parameter_estimation.py.
            if lx > ly:
                lx, ly = ly, lx
                theta = theta + np.pi / 2
            theta = theta % np.pi

            pos_values.append([rx, ry])
            l_values.append([lx, ly])
            theta_values.append(theta)
        else:
            # If fitting fails, add NaNs
            pos_values.append([np.nan, np.nan])
            l_values.append([np.nan, np.nan])
            theta_values.append(np.nan)

    return pos_values, l_values, theta_values


def get_contour_evolution(
    event,
    threshold_factor=0.5,
    max_displacement_threshold=None,
):
    """
    Extract and store the evolution of a single contour, its centroid, and geometric properties in (r, z) coordinates.
    The contours are defined at a level given by threshold_factor times the maximum amplitude of the event.
    Returns None if the maximum frame-to-frame displacement exceeds max_displacement_threshold.

    Parameters
    ----------
    event : xr.Dataset or xr.DataArray
        Input dataset with dimensions (time, x, y), containing 'frames' (2D data),
        'R' (2D radial coordinates), and 'Z' (2D axial coordinates).
    threshold_factor : float, optional
        Factor to determine contour threshold as max(frame) * threshold_factor.
        Default is 0.5.
    max_displacement_threshold : float, optional
        Maximum allowed frame-to-frame COM displacement. If exceeded, returns None.
        If None, no filtering is applied. Must be non-negative if provided.

    Returns
    -------
    xr.Dataset or None
        If max_displacement <= max_displacement_threshold (or threshold is None), returns Dataset containing:
        - 'contours': DataArray of contours in (r, z) coordinates with dims (time, point_idx, coord).
        - 'length': DataArray of contour lengths with dim (time).
        - 'convexity_deficiency': DataArray of convexity deficiencies with dim (time).
        - 'centroid': DataArray of contour center of mass in (r, z) coordinates with dims (time, coord).
        - 'area': DataArray of contour areas with dim (time).
        - 'max_displacement': DataArray of maximum frame-to-frame COM displacement (scalar).
        Returns None if max_displacement > max_displacement_threshold or if fewer than two time points
        with a non-None threshold.
    """
    # Extract time coordinate and data
    time_coords = event.time.values
    frames_da = event.frames if "frames" in event else event
    contours = []
    lengths = []
    convexity_deficiencies = []
    centroids = []
    areas = []
    within_boundaries = []

    # Process each time step
    max_amplitude = np.max(frames_da.values)
    for t in time_coords:
        # Extract 2D frame
        frame = frames_da.sel(time=t).values

        # Find contours using threshold = max_amplitude * threshold_factor
        threshold = max_amplitude * threshold_factor
        contour_list = measure.find_contours(frame, threshold)

        # Select single contour (highest mass if multiple)
        contour = contour_list[0] if contour_list else np.array([])
        if len(contour_list) > 1:
            contour = max(
                contour_list,
                key=lambda c: compute_contour_mass(
                    c, frame, event.R.values, event.Z.values
                ),
            )

        # Convert to (r, z) coordinates
        coords = np.array([]).reshape(0, 2)
        length = 0.0
        convexity_deficiency = 0.0
        centroid = np.array([np.nan, np.nan])
        area = 0.0
        if len(contour) > 0:
            r_values, z_values = indexes_to_coordinates(
                event.R.values, event.Z.values, contour
            )
            coords = np.stack((r_values, z_values), axis=-1)

            # Compute geometric properties
            try:
                polygon = Polygon(coords)
                convex_hull = ConvexHull(coords)
                length = polygon.length
                convexity_deficiency = abs(
                    (convex_hull.volume - polygon.area) / convex_hull.volume
                )
                area = polygon.area
                centroid = np.array([polygon.centroid.x, polygon.centroid.y])
            except (ValueError, ZeroDivisionError):
                # Handle invalid polygons or zero convex hull volume
                length = 0.0
                convexity_deficiency = 0.0
                centroid = np.array([np.nan, np.nan])
                area = 0.0

        contours.append(coords)
        lengths.append(length)
        convexity_deficiencies.append(convexity_deficiency)
        centroids.append(centroid)
        areas.append(area)
        # Evaluated on the index-space contour, before the (r, z) conversion, so
        # the bound is the array itself and needs no assumption about dim order.
        within_boundaries.append(is_within_boundaries(contour, frame.shape))

    # Compute maximum frame-to-frame displacement
    max_displacement = np.nan
    if len(time_coords) >= 2:
        positions = np.array(centroids)  # Shape: (time, 2)
        displacements = np.sqrt(np.sum((positions[1:] - positions[:-1]) ** 2, axis=1))
        if np.any(np.isnan(displacements)):
            max_displacement = np.inf
        else:
            max_displacement = (
                np.max(displacements) if len(displacements) > 0 else np.nan
            )
        if (
            not np.isnan(max_displacement)
            and max_displacement_threshold is not None
            and max_displacement > max_displacement_threshold
        ):
            print(f"Exceeded maximum displacement: {max_displacement:.2f}")
            return None

    # Create ragged array for contours
    max_points = max(len(c) for c in contours) if contours else 1
    contour_data = np.full((len(time_coords), max_points, 2), np.nan)
    for i, contour in enumerate(contours):
        if len(contour) > 0:
            contour_data[i, : len(contour), :] = contour

    pos_values, l_values, theta_values = fit_ellipses_to_contour_ds(contour_data)

    # Create DataArrays
    contours_da = xr.DataArray(
        contour_data,
        dims=("time", "point_idx", "coord"),
        coords={"time": time_coords, "coord": ["r", "z"]},
        attrs={
            "description": "Contours of coherent structure in (r, z) coordinates",
            "algorithm": "skimage.measure.find_contours",
            "threshold": f"max(frame) * {threshold_factor}",
        },
    )

    length_da = xr.DataArray(
        lengths,
        dims=("time",),
        coords={"time": time_coords},
        attrs={"description": "Perimeter length of the contour"},
    )

    convexity_da = xr.DataArray(
        convexity_deficiencies,
        dims=("time",),
        coords={"time": time_coords},
        attrs={
            "description": "Convexity deficiency: abs((convex_area - polygon_area) / convex_area)"
        },
    )

    centroid_da = xr.DataArray(
        centroids,
        dims=("time", "coord"),
        coords={"time": time_coords, "coord": ["r", "z"]},
        attrs={"description": "Geometric centroid of the contour polygon"},
    )

    pos_da = xr.DataArray(
        pos_values,
        dims=("time", "coord"),
        coords={"time": time_coords, "coord": ["r", "z"]},
        attrs={
            "description": "Position values (r, z) computed by fitting an ellipse to the contour at each time"
        },
    )

    size_da = xr.DataArray(
        l_values,
        dims=("time", "coord"),
        coords={"time": time_coords, "coord": ["r", "z"]},
        attrs={
            "description": "Length sizes (lx, ly) computed by fitting an ellipse to the contour at each time"
        },
    )

    theta_da = xr.DataArray(
        theta_values,
        dims=("time",),
        coords={"time": time_coords},
        attrs={
            "description": "Theta angles computed by fitting an ellipse to the contour at each time"
        },
    )

    area_da = xr.DataArray(
        areas,
        dims=("time",),
        coords={"time": time_coords},
        attrs={"description": "Area of the contour polygon in (r, z) coordinates"},
    )

    within_boundaries_da = xr.DataArray(
        np.array(within_boundaries, dtype=bool),
        dims=("time",),
        coords={"time": time_coords},
        attrs={
            "description": "Whether the contour closes strictly inside the array border"
        },
    )

    # Combine into Dataset
    contour_ds = xr.Dataset(
        {
            "contours": contours_da,
            "length": length_da,
            "convexity_deficiency": convexity_da,
            "centroid": centroid_da,
            "within_boundaries": within_boundaries_da,
            "position": pos_da,
            "size": size_da,
            "theta": theta_da,
            "area": area_da,
        }
    )
    contour_ds["max_displacement"] = max_displacement

    return contour_ds


# The contour level read off the conditional average instead of set by hand.
# Moved from fusion_scripts `twodca_manuscript/contour_level.py` at 12b8543.


def zero_lag_frame(average_ds, variable="cond_av"):
    """The (y, x) frame of ``variable`` at the lag closest to zero."""
    field = average_ds[variable].transpose("time", "y", "x")
    return field.values[int(np.abs(field.time.values).argmin())]


def neighbour_ratios(frame, step=1):
    """Values of the peak's 4-neighbours, as fractions of the peak.

    Fewer than four when the peak sits on the array border -- the ratios are averaged,
    so a border pixel simply averages what it has. Border pixels are not excluded: on
    the W7-X puff-1 velocity field they already fail the tracking on their own, 2 of 44
    fitted against 50 of 84 in the interior.

    ``step`` pixels away rather than one: on a view sampled much finer than the
    structure, the nearest neighbours sit at the peak and the level at ~1.
    """
    ny, nx = frame.shape
    y, x = np.unravel_index(np.argmax(frame), frame.shape)
    neighbours = [
        frame[y + dy, x + dx]
        for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step))
        if 0 <= y + dy < ny and 0 <= x + dx < nx
    ]
    return np.array(neighbours) / frame[y, x]


def neighbour_level(average_ds, variable="cond_av", step=1):
    """Contour level for one conditional average: the neighbour mean.

    Takes the zero-lag frame only -- the peak pixel, which is normally the reference
    pixel (``check_max`` makes it brighter than its neighbours, not the frame's
    maximum, and ``check_max=0`` guarantees nothing), and its four nearest neighbours
    ``step`` pixels away -- and returns

        threshold_factor = mean(neighbour values) / peak value

    so the level set is the one that reaches, on average, one step out from the peak.
    It is a lookup, not a search: no optimisation, no scan over levels, and the same
    level is then used at every lag (``get_contour_evolution``'s ``threshold_factor``).
    For a velocity field it is recomputed per pixel, from that pixel's own conditional
    average.

    A hand-set level does not transfer between records: on W7-X, the 0.48 that closes
    the zero-lag contour of one puff leaves a two-lag fit and -1486 m/s on another puff
    of the same record. On C-Mod this rule lands on 0.425 against the hand-set 0.43.
    Against the closure floor (the lowest level whose zero-lag set still closes inside
    the array), the other candidate that adapts, it is both more stable under
    resampling the events and more discriminating between records. The four neighbours
    are not equal, so averaging them is the one arbitrary choice in the rule.

    Deliberately no guards: no closure floor as a lower bound, no minimum number of
    lags. A pixel whose contour opens or whose mask collapses at its own level is bad
    data, and no velocity is the right answer for it.
    """
    return float(neighbour_ratios(zero_lag_frame(average_ds, variable), step).mean())

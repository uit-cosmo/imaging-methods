import xarray as xr
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull


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

    # Create grid of (r, z) points from R, Z
    rows, cols = frame.shape
    r_grid, z_grid = R, Z  # Assume R, Z are 2D arrays
    points = np.vstack((r_grid.ravel(), z_grid.ravel())).T  # Shape: (rows*cols, 2)

    # Check which points are inside the polygon
    mask = np.zeros(len(points), dtype=bool)
    for i, (r, z) in enumerate(points):
        mask[i] = polygon.contains(Point(r, z))

    # Reshape mask to 2D
    mask_2d = mask.reshape(rows, cols)

    # Sum intensity values within the mask
    mass = np.sum(frame[mask_2d])

    return float(mass) if not np.isnan(mass) else 0.0


def indexes_to_coordinates(R, Z, indexes):
    """Convert contour indices to (r, z) coordinates using R, Z grids."""
    dx = R[0, 1] - R[0, 0]
    dy = Z[1, 0] - Z[0, 0]
    r_values = np.min(R) + indexes[:, 1] * dx
    z_values = np.min(Z) + indexes[:, 0] * dy
    return r_values, z_values


def get_contour_evolution(event, threshold_factor=0.5, max_displacement_threshold=None):
    """
    Extract and store the evolution of a single contour, its center of mass, and geometric properties in (r, z) coordinates.
    Returns None if the maximum frame-to-frame COM displacement exceeds max_displacement_threshold.

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
        - 'center_of_mass': DataArray of contour center of mass in (r, z) coordinates with dims (time, coord).
        - 'area': DataArray of contour areas with dim (time).
        - 'max_displacement': DataArray of maximum frame-to-frame COM displacement (scalar).
        Returns None if max_displacement > max_displacement_threshold or if fewer than two time points
        with a non-None threshold.
    """
    # Input validation
    if not isinstance(event, (xr.Dataset, xr.DataArray)):
        raise ValueError("Input 'event' must be an xarray Dataset or DataArray")
    if "frames" not in event and not isinstance(event, xr.DataArray):
        raise ValueError("Input must contain 'frames' or be a DataArray")
    if "R" not in event.coords or "Z" not in event.coords:
        raise ValueError("Input must include 'R' and 'Z' coordinates")
    if max_displacement_threshold is not None and max_displacement_threshold < 0:
        raise ValueError("max_displacement_threshold must be non-negative")
    # Extract time coordinate and data
    time_coords = event.time.values
    contours = []
    lengths = []
    convexity_deficiencies = []
    centers_of_mass = []
    areas = []

    # Process each time step
    for t in time_coords:
        # Extract 2D frame
        frame = (
            event.frames.sel(time=t).values
            if "frames" in event
            else event.sel(time=t).values
        )

        # Find contours using threshold = max(frame) * threshold_factor
        threshold = np.max(frame) * threshold_factor
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
        com = np.array([np.nan, np.nan])
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
                com = np.array([polygon.centroid.x, polygon.centroid.y])
                area = polygon.area
            except (ValueError, ZeroDivisionError):
                # Handle invalid polygons or zero convex hull volume
                length = 0.0
                convexity_deficiency = 0.0
                com = np.array([np.nan, np.nan])
                area = 0.0

        contours.append(coords)
        lengths.append(length)
        convexity_deficiencies.append(convexity_deficiency)
        centers_of_mass.append(com)
        areas.append(area)

    # Check if max_displacement exceeds threshold or insufficient time points
    if max_displacement_threshold is not None:
        # Compute maximum frame-to-frame displacement
        max_displacement = np.nan
        if len(time_coords) >= 2:
            com_values = np.array(centers_of_mass)  # Shape: (time, 2)
            displacements = np.sqrt(
                np.sum((com_values[1:] - com_values[:-1]) ** 2, axis=1)
            )
            valid_displacements = displacements[~np.isnan(displacements)]
            max_displacement = (
                np.max(valid_displacements) if len(valid_displacements) > 0 else np.nan
            )
        if len(time_coords) < 2 or (
            not np.isnan(max_displacement)
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

    com_da = xr.DataArray(
        centers_of_mass,
        dims=("time", "coord"),
        coords={"time": time_coords, "coord": ["r", "z"]},
        attrs={"description": "Center of mass of the contour in (r, z) coordinates"},
    )

    area_da = xr.DataArray(
        areas,
        dims=("time",),
        coords={"time": time_coords},
        attrs={"description": "Area of the contour polygon in (r, z) coordinates"},
    )

    # Combine into Dataset
    return xr.Dataset(
        {
            "contours": contours_da,
            "length": length_da,
            "convexity_deficiency": convexity_da,
            "center_of_mass": com_da,
            "area": area_da,
        }
    )


from scipy.ndimage import gaussian_filter1d


def get_contour_velocity(com_da, sigma=1.0):
    """
    Compute the velocity of a structure from its center of mass coordinates with Gaussian smoothing.

    Parameters
    ----------
    com_da : xr.DataArray
        Center of mass coordinates with dimensions (time, coord), where coord=["r", "z"].
        Typically obtained from get_contour_evolution output as ds.center_of_mass.
    sigma : float, optional
        Standard deviation for Gaussian smoothing kernel (in time steps). Default is 1.0.

    Returns
    -------
    xr.DataArray
        Velocity of the center of mass in (r, z) coordinates with dimensions (time, coord).
        Units are com_da units per time unit (e.g., cm/s if com_da is in cm and time in seconds).
    """
    # Input validation
    if not isinstance(com_da, xr.DataArray):
        raise ValueError("Input 'com_da' must be an xarray DataArray")
    if com_da.dims != ("time", "coord") or com_da.shape[-1] != 2:
        raise ValueError(
            "Input must have dimensions (time, coord) with coord=['r', 'z']"
        )
    if "time" not in com_da.coords or "coord" not in com_da.coords:
        raise ValueError("Input must include 'time' and 'coord' coordinates")
    if not np.array_equal(com_da.coord.values, ["r", "z"]):
        raise ValueError("Input coord must be ['r', 'z']")
    if sigma < 0:
        raise ValueError("Smoothing sigma must be non-negative")

    # Extract time and COM data
    time = com_da.time.values
    com = com_da.values  # Shape: (time, 2) for (r, z)
    n_times = len(time)

    # Handle insufficient time points
    if n_times < 2:
        raise ValueError("At least two time points are required to compute velocity")

    # Smooth COM data (r and z separately)
    com_smooth = np.zeros_like(com)
    for i in range(com.shape[1]):  # Loop over r, z
        # Handle NaNs by interpolating before smoothing
        mask = ~np.isnan(com[:, i])
        if np.sum(mask) > 1:  # Need at least 2 non-NaN points
            com_smooth[:, i] = gaussian_filter1d(
                np.interp(np.arange(n_times), np.where(mask)[0], com[mask, i]),
                sigma=sigma,
            )
        else:
            com_smooth[:, i] = com[:, i]  # Copy original if insufficient data

    # Initialize velocity array
    velocity = np.zeros_like(com_smooth)  # Shape: (time, 2)

    # Forward difference for first point
    dt = time[1] - time[0]
    velocity[0] = (
        (com_smooth[1] - com_smooth[0]) / dt
        if not np.any(np.isnan(com_smooth[0]))
        else np.array([np.nan, np.nan])
    )

    # Central difference for interior points
    for i in range(1, n_times - 1):
        if np.any(np.isnan(com_smooth[i])):
            velocity[i] = np.array([np.nan, np.nan])
        else:
            velocity[i] = (com_smooth[i + 1] - com_smooth[i - 1]) / (2 * dt)

    # Backward difference for last point
    velocity[-1] = (
        (com_smooth[-1] - com_smooth[-2]) / dt
        if not np.any(np.isnan(com_smooth[-1]))
        else np.array([np.nan, np.nan])
    )

    # Create velocity DataArray
    velocity_da = xr.DataArray(
        velocity,
        dims=("time", "coord"),
        coords={"time": time, "coord": ["r", "z"]},
        attrs={
            "description": "Velocity of the contour center of mass in (r, z) coordinates",
            "method": "Finite difference (central for interior, forward/backward for endpoints) on Gaussian-smoothed COM data",
            "smoothing": f"Gaussian filter with sigma={sigma} time steps",
            "units": "Same as com_da units per time unit",
        },
    )

    return velocity_da

import xarray as xr
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from skimage.measure import EllipseModel


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
            if lx > ly:
                lx, ly = ly, lx
                theta = theta - np.pi/4 if theta > np.pi/4 else theta + np.pi/4

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
    contours = []
    lengths = []
    convexity_deficiencies = []
    centroids = []
    areas = []

    # Process each time step
    max_amplitude = np.max(event.values)
    for t in time_coords:
        # Extract 2D frame
        frame = (
            event.frames.sel(time=t).values
            if "frames" in event
            else event.sel(time=t).values
        )

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

    # Combine into Dataset
    contour_ds = xr.Dataset(
        {
            "contours": contours_da,
            "length": length_da,
            "convexity_deficiency": convexity_da,
            "centroid": centroid_da,
            "position": pos_da,
            "size": size_da,
            "theta": theta_da,
            "area": area_da,
        }
    )
    contour_ds["max_displacement"] = max_displacement

    return contour_ds

import xarray as xr
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from scipy.signal import windows
from matplotlib.path import Path as MplPath


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


def get_contour_evolution(
    event,
    threshold_factor=0.5,
    max_displacement_threshold=None,
    com_method="centroid",
):
    """
    Extract and store the evolution of a single contour, its center of mass, and geometric properties in (r, z) coordinates.
    The contours are defined at a level given by threshold_factor times the maximum amplitude of the event.
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
    com_method : str, optional
        Method to compute the center of mass. Options:
        - 'centroid': Geometric centroid of the contour polygon (default).
        - 'com': Intensity-weighted center of mass inside the contour.
        - 'global': Intensity-weighted center of mass over the full frame.

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
    if com_method not in ["centroid", "com", "global"]:
        raise ValueError("com_method must be 'centroid', 'com', or 'global'")

    # Extract time coordinate and data
    time_coords = event.time.values
    contours = []
    lengths = []
    convexity_deficiencies = []
    centers_of_mass = []
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
                area = polygon.area

                if com_method == "centroid":
                    com = np.array([polygon.centroid.x, polygon.centroid.y])
                elif com_method == "com":
                    # Compute intensity-weighted CoM inside contour
                    ny, nx = frame.shape
                    yy, xx = np.meshgrid(range(ny), range(nx), indexing="ij")
                    points = np.c_[yy.ravel(), xx.ravel()]  # (ny*nx, 2) [y, x]
                    path = MplPath(contour, closed=True)
                    mask_flat = path.contains_points(points)
                    mask = mask_flat.reshape((ny, nx))

                    weights = frame * mask
                    total_mass = weights.sum()
                    if total_mass > 0:
                        com_r = (weights * event.R.values).sum() / total_mass
                        com_z = (weights * event.Z.values).sum() / total_mass
                        com = np.array([com_r, com_z])
                    else:
                        com = np.array([np.nan, np.nan])
                elif com_method == "global":
                    # Compute intensity-weighted CoM over full frame
                    weights = frame
                    total_mass = weights.sum()
                    if total_mass > 0:
                        com_r = (weights * event.R.values).sum() / total_mass
                        com_z = (weights * event.Z.values).sum() / total_mass
                        com = np.array([com_r, com_z])
                    else:
                        com = np.array([np.nan, np.nan])

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

    # Compute maximum frame-to-frame displacement
    max_displacement = np.nan
    if len(time_coords) >= 2:
        com_values = np.array(centers_of_mass)  # Shape: (time, 2)
        displacements = np.sqrt(np.sum((com_values[1:] - com_values[:-1]) ** 2, axis=1))
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

    if com_method == "centroid":
        com_description = "Geometric centroid of the contour polygon"
    elif com_method == "com":
        com_description = "Intensity-weighted center of mass inside the contour"
    else:  # global
        com_description = "Intensity-weighted center of mass over the full frame"

    com_da = xr.DataArray(
        centers_of_mass,
        dims=("time", "coord"),
        coords={"time": time_coords, "coord": ["r", "z"]},
        attrs={"description": com_description},
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
            "center_of_mass": com_da,
            "area": area_da,
        }
    )
    contour_ds["max_displacement"] = max_displacement

    return contour_ds


def get_contour_velocity(com_da, window_size=3, window_type="boxcar"):
    """
    Compute the velocity of center of mass positions using a specified smoothing window from scipy.signal.windows.

    Parameters:
    - com_da (xr.DataArray): Center of mass positions with dims ('time', 'coord') and coords 'r', 'z'.
    - window_size (int): Size of the smoothing window (default: 3).
    - window_type (str): Type of smoothing window ('boxcar', 'gaussian', 'hamming', 'blackman', 'triang').

    Returns:
    - velocity_da (xr.DataArray): Velocity with dims ('time', 'coord'), cropped to valid time points.
    """
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError("window_size must be a positive integer")
    if window_type not in ["boxcar", "gaussian", "hamming", "blackman", "triang"]:
        raise ValueError(
            "window_type must be 'boxcar', 'gaussian', 'hamming', 'blackman', or 'triang'"
        )
    if len(com_da.time) < 2:
        raise ValueError("At least two time points are required")

    # Interpolate NaNs
    com_interp = com_da.interpolate_na(
        dim="time", method="nearest", fill_value="extrapolate"
    )
    com_np = com_interp.values  # Shape: (n_times, 2)

    # Define window parameters
    half_window = window_size // 2
    n_times = len(com_da.time)
    start = half_window + 1
    end = n_times - half_window - 1

    # Check if there are enough points
    if start >= end:
        raise ValueError(
            f"Not enough time points to compute velocity with window_size={window_size}"
        )

    # Generate window using scipy.signal.windows
    if window_type == "gaussian":
        window = windows.gaussian(window_size, std=window_size / 4, sym=True)
    else:
        window = getattr(windows, window_type)(window_size, sym=True)
    window /= window.sum()  # Normalize

    # Compute moving average
    com_smooth = np.array(
        [np.convolve(com_np[:, i], window, mode="same") for i in range(2)]
    ).T

    # Compute time step
    dt = float(com_da.time[1] - com_da.time[0])

    # Compute velocity using central differences
    valid_times = com_da.time[start:end]
    velocity_values = (
        com_smooth[start + 1 : end + 1, :] - com_smooth[start - 1 : end - 1, :]
    ) / (2 * dt)

    # Create output DataArray
    velocity_da = xr.DataArray(
        velocity_values,
        dims=("time", "coord"),
        coords={"time": valid_times, "coord": com_da.coord},
        attrs={
            "description": "Velocity of the contour center of mass in (r, z) coordinates",
            "method": "Finite difference (central) on smoothed COM data",
            "smoothing": f"{window_type.capitalize()} window with window_size={window_size} time steps",
            "units": "Same as com_da units per time unit",
        },
    )

    return velocity_da


def get_average_velocity_for_near_com(average_ds, contour_ds, velocity_ds, distance):
    refx, refy = average_ds["refx"].item(), average_ds["refy"].item()

    distances_vector = contour_ds.center_of_mass.values - [
        average_ds.R.isel(x=refx, y=refy).item(),
        average_ds.Z.isel(x=refx, y=refy).item(),
    ]
    distances = np.sqrt((distances_vector**2).sum(axis=1))
    mask = distances < distance
    valid_times = contour_ds.time[mask]  # DataArray with wanted times
    common_times = valid_times[valid_times.isin(velocity_ds.time)]

    v_c, w_c = velocity_ds.sel(time=common_times).mean(dim="time", skipna=True).values

    return v_c, w_c

import numpy as np
import xarray as xr

__all__ = [
    "get_physical_pos",
    "compute_maximum_trajectory_da",
    "find_maximum_for_frame",
    "parabolic_2d_interp",
    "find_maximum_interpolate",
    "get_maximum_time",
]


def get_physical_pos(
    da: xr.DataArray,
    x_idx: int,
    y_idx: int,
    sub_x: float | None = None,
    sub_y: float | None = None,
) -> float:
    """Get position from da at (sub_x, sub_y) or discrete (x_idx, y_idx)."""
    if sub_x is None and sub_y is None:
        # Discrete isel
        isel_kwargs = {}
        if "x" in da.dims:
            isel_kwargs["x"] = x_idx
        if "y" in da.dims:
            isel_kwargs["y"] = y_idx
        if not isel_kwargs:
            raise ValueError(f"Invalid dimensions for {da.name}: {da.dims}")
        return float(da.isel(**isel_kwargs).values)
    else:
        # Sub-pixel interp
        interp_kwargs = {"method": "linear"}
        interp_dims = {}
        if "x" in da.dims:
            interp_dims["x"] = sub_x if sub_x is not None else x_idx
        if "y" in da.dims:
            interp_dims["y"] = sub_y if sub_y is not None else y_idx
        if not interp_dims:
            raise ValueError(f"Invalid dimensions for {da.name}: {da.dims}")
        return float(da.interp(interp_dims, **interp_kwargs).values)


def compute_maximum_trajectory_da(
    average_ds: xr.Dataset,
    variable="cond_av",
    method="parabolic",
    min_intensity: float = 0.0,
) -> xr.DataArray:
    """
    Compute the sub-pixel trajectory of the maximum intensity in each time frame
    of `cond_av` from 2DCA, returning an xarray.DataArray in physical R-Z coordinates.

    Parameters
    ----------
    average_ds : xr.Dataset
        Output from `find_events_and_2dca` containing `cond_av` (time, x, y),
        and fields `R` and `Z` for physical coordinates (dims: ('x',), ('y',), or ('x', 'y')).
    method : {'parabolic'}, optional
        Sub-pixel interpolation method. Only 'parabolic' is supported.
    min_intensity : float, optional
        Frames with max intensity below this are set to NaN.

    Returns
    -------
    xr.DataArray
        Trajectory of maximum intensity with:
        - dims: ('time', 'coord')
        - coords: 'time' (relative), 'coord' = ['R', 'Z']
        - values: physical (R, Z) positions
        - attrs: method, number of events

    Raises
    ------
    ValueError
        If input is invalid or missing R/Z fields.
    """
    if "cond_av" not in average_ds:
        raise ValueError("average_ds must contain 'cond_av' DataArray")

    data = average_ds[variable]
    if data.ndim != 3 or set(data.dims) != {"time", "x", "y"}:
        raise ValueError("cond_av must have dimensions (time, x, y)")

    try:
        R_da = average_ds["R"]
        Z_da = average_ds["Z"]
    except KeyError as e:
        raise ValueError(
            "average_ds must contain 'R' and 'Z' fields for physical coordinates"
        ) from e

    time_coords = data["time"].values

    # Compute raw trajectory
    raw_trajectory = np.array(
        [
            find_maximum_for_frame(
                data.isel(time=it),
                R_da,
                Z_da,
                min_intensity=min_intensity,
                method=method,
            )
            for it in range(len(time_coords))
        ]
    )  # Shape: (n_times, 2)

    # Interpolate NaNs (nearest + extrapolate)
    traj_da = xr.DataArray(
        raw_trajectory,
        dims=("time", "coord"),
        coords={"time": time_coords, "coord": ["R", "Z"]},
    )
    final_traj = traj_da.interpolate_na(
        dim="time", method="nearest", fill_value="extrapolate"
    )

    # Add attributes
    final_traj.attrs = {
        "description": "Sub-pixel trajectory of maximum intensity in cond_av (physical R-Z coordinates)",
        "method": "2D parabolic interpolation around discrete maximum",
        "min_intensity_threshold": min_intensity,
        "refx": int(average_ds.refx),
        "refy": int(average_ds.refy),
        "number_events": int(average_ds.number_events),
        "units_R": "same as input R units",
        "units_Z": "same as input Z units",
    }

    return final_traj


def find_maximum_for_frame(
    frame_da: xr.DataArray, R_da, Z_da, min_intensity: float = 0, method="parabolic"
):
    if method == "parabolic":
        ind_x, ind_y = parabolic_2d_interp(frame_da, min_intensity)
    else:
        raise ValueError("Unsupported method {}".format(method))

    R = float(R_da.interp({"x": ind_x, "y": ind_y}, method="linear").values)
    Z = float(Z_da.interp({"x": ind_x, "y": ind_y}, method="linear").values)
    return R, Z


def parabolic_2d_interp(
    frame_da: xr.DataArray, min_intensity: float = 0
) -> tuple[float, float]:
    """Return sub-pixel (R, Z) of maximum using 2D parabolic fit in index space."""
    # Find discrete maximum position
    flat_argmax = frame_da.values.argmax()
    y_idx, x_idx = np.unravel_index(
        flat_argmax, (frame_da.sizes["y"], frame_da.sizes["x"])
    )
    val_center = float(frame_da.isel(x=x_idx, y=y_idx).values)

    if val_center < min_intensity:
        return np.nan, np.nan

    nx = frame_da.sizes["x"]
    ny = frame_da.sizes["y"]

    # Edge case: max on boundary → return discrete position
    if not (1 <= x_idx < nx - 1 and 1 <= y_idx < ny - 1):
        return x_idx, y_idx

    # Select 3x3 neighborhood using isel
    nb_da = frame_da.isel(x=slice(x_idx - 1, x_idx + 2), y=slice(y_idx - 1, y_idx + 2))

    # Additional check: if neighborhood not 3x3 (small grid), return discrete
    if nb_da.sizes["x"] != 3 or nb_da.sizes["y"] != 3:
        return x_idx, y_idx

    # Parabolic fit in x (along x, fixed middle y). Only shift if the center is
    # a strict local max (guarantees a finite, well-posed parabola).
    fx_m1 = float(nb_da.isel(y=1, x=0).values)
    fx_0 = float(nb_da.isel(y=1, x=1).values)
    fx_p1 = float(nb_da.isel(y=1, x=2).values)
    if fx_0 > fx_m1 and fx_0 > fx_p1:
        delta_x = 0.5 * (fx_m1 - fx_p1) / (fx_m1 - 2 * fx_0 + fx_p1)
    else:
        delta_x = 0.0

    # Parabolic fit in y (along y, fixed middle x)
    fy_m1 = float(nb_da.isel(y=0, x=1).values)
    fy_0 = float(nb_da.isel(y=1, x=1).values)
    fy_p1 = float(nb_da.isel(y=2, x=1).values)
    if fy_0 > fy_m1 and fy_0 > fy_p1:
        delta_y = 0.5 * (fy_m1 - fy_p1) / (fy_m1 - 2 * fy_0 + fy_p1)
    else:
        delta_y = 0.0

    # The sub-pixel shift can never legitimately exceed half a pixel
    sub_x = x_idx + float(np.clip(delta_x, -0.5, 0.5))
    sub_y = y_idx + float(np.clip(delta_y, -0.5, 0.5))
    return sub_x, sub_y


# When an event peaks at a pixel: the maximum of a spline through its lags.
# Moved from fusion_scripts `density_scan/peak_time.py` at 12b8543.


def find_maximum_interpolate(x, y):
    """The maximum of a quartic interpolating spline through ``(x, y)``: ``(x_max,
    y_max)``, among the roots of its derivative and the two ends of ``x``. Warns when
    the maximum is at an end.

    A copy of velocity_estimation's private
    ``time_delay_estimation._find_maximum_interpolate``, the same search and warning,
    until velocity_estimation makes it public and this calls it."""
    from scipy.interpolate import InterpolatedUnivariateSpline
    import warnings

    # Taking the derivative and finding the roots only work if the spline degree is at least 4.
    spline = InterpolatedUnivariateSpline(x, y, k=4)
    possible_maxima = spline.derivative().roots()
    possible_maxima = np.append(
        possible_maxima, (x[0], x[-1])
    )  # also check the endpoints of the interval
    values = spline(possible_maxima)

    max_index = np.argmax(values)
    max_time = possible_maxima[max_index]
    if max_time == x[0] or max_time == x[-1]:
        warnings.warn(
            "Maximization on interpolation yielded a maximum in the boundary!"
        )

    return max_time, spline(max_time)


def get_maximum_time(e, refx=None, refy=None):
    """The lag at which the event ``e`` peaks at pixel (refx, refy).

    ``e`` is a DataArray over time, y and x with ``R`` among its coordinates, such as a
    conditional average's ``cond_av``; the pixel defaults to ``e["refx"]`` and
    ``e["refy"]``. The lag is the maximum of a quartic spline through the pixel's series
    (``find_maximum_interpolate``). Returns None for a pixel outside the array, and
    warns when the maximum is at the edge of the window.
    """
    import warnings

    if refx is None or refy is None:
        refx, refy = int(e["refx"].item()), int(e["refy"].item())

    is_in_boundaries = 0 <= refx < e.R.sizes["x"] and 0 <= refy < e.R.sizes["y"]
    if not is_in_boundaries:
        return None
    times, data = e.time, e.isel(x=refx, y=refy)

    tau, _ = find_maximum_interpolate(times, data)
    if tau <= np.min(times) or tau >= np.max(times):
        warnings.warn(
            "Time delay found at the window edge, consider running 2DCA with a larger window"
        )
    return tau

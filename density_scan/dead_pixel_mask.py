import xarray as xr

mask = [
    [True, True, False, True, False, False, True, True, True],
    [False, False, False, False, False, False, False, True, True],
    [True, False, False, False, True, False, False, False, True],
    [False, False, False, True, False, False, False, False, False],
    [True, False, False, False, False, False, False, False, True],
    [False, False, False, False, False, False, False, False, True],
    [False, False, False, False, False, False, False, True, False],
    [False, True, False, True, False, False, False, False, True],
    [True, True, False, False, False, False, False, False, False],
    [False, False, True, False, False, False, False, False, False],
]


def get_dead_pixel_mask():
    return xr.DataArray(
        mask[::-1],
        dims=["y", "x"],
        coords={
            "y": range(10),  # y from 0 (top) to 9 (bottom)
            "x": range(9),  # x from 0 to 9
        },
    )

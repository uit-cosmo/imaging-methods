import matplotlib.style
import xarray as xr
import numpy as np
import phantom.data_preprocessing
from numpy.testing import assert_array_equal

from phantom import interpolate_nans_3d
import cosmoplots as cp
import matplotlib.pyplot as plt

from tests.test_data_preprocessing import ds_interpolated

shot = 1160616009
ds_raw = xr.load_dataset("../data/apd_{}.nc".format(shot))

ds_raw = ds_raw.isel(time=slice(0, 10))
ds_original = ds_raw.copy(deep=True)
ds_interpolated = interpolate_nans_3d(ds_raw)

matplotlib.style.use("cosmoplots.default")
fig, ax = cp.figure_multiple_rows_columns(1, 2)
ax[0].imshow(ds_original.frames.isel(time=5).values)
ax[1].imshow(ds_interpolated.frames.isel(time=5).values)

ax[0].set_title("Raw")
ax[1].set_title("Interpolated")

fig.savefig("test_interpolation.eps")
plt.show()

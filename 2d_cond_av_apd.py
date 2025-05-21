from phantom.utils import *
from phantom import fit_psd
import matplotlib.pyplot as plt
from phantom.contours import *

ds = xr.open_dataset("data/apd_1160616027_preprocessed.nc")
ds = ds.isel(time=slice(2000, -2000))


refx, refy = 6, 5

fig, ax = plt.subplots()

taud, lam = fit_psd(
    ds.frames.isel(x=refx, y=refy).values, get_dt(ds), nperseg=10**4, ax=ax
)

plt.show()

from phantom.show_data import *
from phantom.utils import *
from phantom.power_spectral_density import *
from velocity_estimation.correlation import corr_fun
from scipy.optimize import minimize
from scipy import signal

shot = 1160616025

ds = get_sample_data(shot, 0.001)
# ds = xr.open_dataset("data.nc")
refx, refy = 6, 5

fig, ax = plt.subplots()

data_series = ds.frames.isel(x=refx, y=refy).values
tau, lam = fit_psd(data_series, dt=get_dt(ds), ax=ax)

plt.show()

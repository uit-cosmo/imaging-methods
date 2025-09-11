import numpy as np

from phantom.utils import *
import matplotlib.pyplot as plt
from phantom.contours import *
from phantom import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cosmoplots as cp
import velocity_estimation as ve

shot = 1160616027
manager = PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")
ds = manager.read_shot_data(shot, preprocessed=True)

refx, refy = 6, 6

events, average_ds = find_events_and_2dca(
    ds, refx, refy, threshold=2, check_max=1, window_size=60, single_counting=True
)

contour_ds = get_contour_evolution(
    average_ds.cond_av,
    0.3,
    max_displacement_threshold=None,
)
velocity_ds = get_contour_velocity(
    contour_ds.center_of_mass,
    window_size=3,
)
v_c, w_c = velocity_ds.isel(time=slice(10, -10)).mean(dim="time", skipna=True).values

eo = ve.EstimationOptions()
eo.cc_options.cc_window = 60 * get_dt(ds)
pd = ve.estimate_velocities_for_pixel(refx, refy, ve.CModImagingDataInterface(ds))
v_tde, w_tde = pd.vx, pd.vy

print(v_c, w_c)
print(v_tde, w_tde)

from phantom.contours import get_contour_evolution, get_contour_velocity
from phantom.show_data import *
from phantom.cond_av import *
from phantom.data_preprocessing import load_data_and_preprocess
import matplotlib.pyplot as plt

shot = 1160616025
ds = load_data_and_preprocess(shot, 0.005)
ds = ds.isel(x=slice(4, 9), y=slice(3, 8))
# ds.to_netcdf("ds_imode_long.nc")


refx, refy = 2, 2
events, average, std = find_events_and_2dca(
    ds, refx, refy, threshold=2, check_max=0, window_size=20, single_counting=True
)
fig, ax = plt.subplots()

for e in events:
    contours_ds = get_contour_evolution(e, 0.75, max_displacement_threshold=None)
    if contours_ds["max_displacement"] < 0.5:
        velocity = get_contour_velocity(contours_ds.center_of_mass, sigma=3)
        ax.plot(velocity.time.values, velocity.values[:, 0] / 100)
        show_movie_with_contours(
            e,
            refx,
            refy,
            contours_ds,
            variable="frames",
            interpolation=None,
            gif_name="e{}_local.gif".format(e["event_id"].item()),
            show=False,
        )

plt.show()

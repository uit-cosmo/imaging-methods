import imaging_methods as im
import matplotlib.pyplot as plt
import cosmoplots as cp
import xarray as xr
from imaging_methods import movie_dataset, show_movie_with_contours

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

# If the file does not exist, it is created by the synthetic_blob_2dca.py script
file_name = "synthetic_blob_2dca.nc"
ds = xr.open_dataset(file_name)

dt = im.get_dt(ds)
dr = im.get_dr(ds)

movie_dataset(
    ds.sel(time=slice(500, 520)),
    gif_name="realization.gif",
    show=False,
    lims=(0, 1),
    normalize=True,
    interpolation=None,
)

mp = im.get_default_synthetic_method_params()
events, average_ds = im.find_events_and_2dca(ds, mp.two_dca)

contour_ca = im.get_contour_evolution(
    average_ds.cond_av,
    mp.contouring.threshold_factor,
    max_displacement_threshold=None,
)

show_movie_with_contours(
    average_ds,
    contour_ca,
    variable="cond_av",
    lims=(0, 1),
    gif_name="contours.gif",
    interpolation=None,
    show=False,
)

from imaging_methods import *
import cosmoplots as cp

manager = GPIDataAccessor(
    "/home/sosno/Git/experimental_database/plasma_discharges.json"
)

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

shot = 1160616011
window = 5e-5

ds = manager.read_shot_data(shot)
middle_time = ds.time.values.mean() + window * 3 / 2
ds_short = ds.sel(time=slice(middle_time - window / 2, middle_time + window / 2))


movie_dataset(
    ds_short,
    variable="frames",
    lims=(0, ds_short.frames.max().item()),
    gif_name="apd_data.gif",
    interpolation=None,
)

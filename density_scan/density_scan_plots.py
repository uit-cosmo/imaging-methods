import phantom as ph
import matplotlib.pyplot as plt
import xarray as xr
import velocity_estimation as ve
import os

manager = ph.PlasmaDischargeManager()
manager.load_from_json("plasma_discharges.json")

analysis = True
plot_movies = False
plot_velocities = False

refx, refy = 6, 6

def get_average(shot):
    file_name = os.path.join("averages", f"average_ds_{shot}.nc")
    return xr.open_dataset(file_name)

if plot_movies:
    for shot in manager.get_shot_list():
        average_ds = get_average(shot)
        gif_name = "2d_cond_av_average_{}.gif".format(shot)
        ph.show_movie(
            average_ds,
            variable="cond_av",
            #        lims=(0, np.max(average.frames.values)),
            gif_name=gif_name,
            show=False,
        )


if analysis:
    results = ph.ScanResults()
    use_contouring = False
    for shot in manager.get_shot_list():
        confinement_more = manager.get_discharge_by_shot(shot).confinement_mode
        is_hmode = confinement_more == "EDA-H" or confinement_more == "ELM-free-H"
        print("Working on shot {}".format(shot))
        average_ds = get_average(shot)
        gpi_ds = manager.read_shot_data(shot, data_folder="../data")

        if use_contouring:
            contour_ds = ph.get_contour_evolution(
                average_ds.cond_av, 0.75, max_displacement_threshold=None
            )
            velocity_ds = ph.get_contour_velocity(contour_ds.center_of_mass, sigma=3)
            v, w = (
                velocity_ds.isel(time=slice(10, -10))
                .mean(dim="time", skipna=True)
                .values
            )
            v, w = v / 100, w / 100

            area = contour_ds.area.mean(dim="time").item()
            area = area / 100**2
            lx, ly, theta = area, 0, 0

            ph.show_movie_with_contours(
                average_ds,
                refx,
                refy,
                contour_ds,
                "cond_av",
                gif_name="average_contour_{}.gif".format(shot),
                interpolation="spline16",
                show=False,
            )
        else:
            v, w = ph.get_3tde_velocities(average_ds.cond_av, average_ds["refx"], average_ds["refy"])
            v, w = v / 100, w / 100

            fig, ax = plt.subplots()
            lx, ly, theta = ph.plot_event_with_fit(
                average_ds.cond_av, refx, refy, ax, "average_fig_{}.png".format(shot)
            )
            lx, ly = lx / 100, ly / 100
            fig.clf()

        fig, ax = plt.subplots()

        taud, lam, freqs = ph.DurationTimeEstimator(
            ph.SecondOrderStatistic.PSD, ph.Analytics.TwoSided
        ).plot_and_fit(
            gpi_ds.frames.isel(x=refx, y=refy).values,
            ph.get_dt(average_ds),
            ax,
            cutoff=1e6,
            nperseg=1e3,
        )

        plt.savefig("psd_{}.png".format(shot), bbox_inches="tight")
        fig.clf()

        results.add_shot(
            ph.ShotAnalysis(
                v,
                w,
                lx,
                ly,
                theta,
                taud,
                lam,
                manager.get_discharge_by_shot(shot),
            )
        )
    #    results.to_json("results_contouring.json")
    results.print_summary()

if plot_velocities:
    fig, ax = plt.subplots()
    for shot in manager.get_shot_list():
        print("Working on shot {}".format(shot))
        ds = manager.read_shot_data(shot, None)
        average_ds = get_average(shot)
        contour_ds = ph.get_contour_evolution(
            average_ds.cond_av, 0.75, max_displacement_threshold=None
        )
        velocity_ds = ph.get_contour_velocity(contour_ds.center_of_mass, sigma=1)
        ax.plot(
            velocity_ds.time.values, velocity_ds.values[:, 0], label="{}".format(shot)
        )

    plt.savefig("velocity_evolution.png".format(shot), bbox_inches="tight")
    fig.show()

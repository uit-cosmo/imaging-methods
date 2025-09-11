import numpy as np
import argparse
import sys
import xarray as xr
from phantom.show_data import *
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot event")
    parser.add_argument("event", type=int, help="Event id")

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle invalid arguments gracefully
        print("Error: Please provide a valid integer argument")
        sys.exit(1)

    # Validate the integer input
    event_id = args.event
    if event_id <= 0:
        print("Error: Number of points must be positive")
        sys.exit(1)

    shot = 1160616018
    event = xr.open_dataset("tmp/event_{}_{}.nc".format(shot, event_id))

    fig, ax = plt.subplots()
    R = event.R.isel(x=event["refx"].item(), y=event["refy"].item())
    Z = event.Z.isel(x=event["refx"].item(), y=event["refy"].item())
    ax.scatter(R, Z, color="black")
    show_movie(
        event,
        variable="frames",
        gif_name="output.gif".format(shot),
        fps=60,
        interpolation="spline16",
        lims=(0, np.max(event.frames.values)),
        fig=fig,
        ax=ax,
    )
    os.system("gifsicle -i output.gif -O3 --colors 32 --lossy=150 -o output.gif")

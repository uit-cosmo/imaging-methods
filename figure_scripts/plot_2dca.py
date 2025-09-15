import numpy as np
import xarray as xr
import argparse
import sys
from imaging_methods import *
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average")
    parser.add_argument("shot", type=int, help="Shot number")
    parser.add_argument("refx", type=int, help="refx")
    parser.add_argument("refy", type=int, help="refy")

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle invalid arguments gracefully
        print("Error: Please provide a valid integer argument")
        sys.exit(1)

    # Validate the integer input
    shot = args.shot
    refx = args.refx
    refy = args.refy

    average = xr.open_dataset(
        "density_scan/averages/average_ds_{}_{}{}.nc".format(shot, refx, refy)
    )

    contour_ds = get_contour_evolution(
        average.cond_av,
        0.3,
        max_displacement_threshold=None,
    )
    output_name = "2dca_{}_{}{}.gif".format(shot, refx, refy)
    show_movie_with_contours(
        average,
        contour_ds,
        "cond_av",
        lims=(0, 3),
        gif_name=output_name,
        interpolation="spline16",
        show=True,
    )

    os.system(
        "gifsicle -i {} -O3 --colors 32 --lossy=150 -o {}".format(
            output_name, output_name
        )
    )

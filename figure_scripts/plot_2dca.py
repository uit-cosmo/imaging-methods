import argparse
import sys
from imaging_methods import *
import os
import cosmoplots as cp

params = plt.rcParams
cp.set_rcparams_dynamo(params, 1)
plt.rcParams.update(params)

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
    manager = GPIDataAccessor(
        "/home/sosno/Git/experimental_database/plasma_discharges.json"
    )

    movie_2dca_with_contours(shot, refx, refy, run_2dca=True)

    output_name = "2dcc_{}_{}{}.gif".format(shot, refx, refy)
    # os.system(
    #   "gifsicle -i {} -O3 --colors 32 --lossy=150 -o {}".format(
    #       output_name, output_name
    #   )
    # )

import argparse
import sys
from phantom.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot blob")
    parser.add_argument("lx", type=float, help="lx")
    parser.add_argument("ly", type=float, help="ly")
    parser.add_argument("theta", type=float, help="theta")

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Handle invalid arguments gracefully
        print("Error: Please provide a valid integer argument")
        sys.exit(1)

    # Validate the integer input
    lx = args.lx
    ly = args.ly
    theta = args.theta

    L = max(lx, ly)
    xvals = np.linspace(-3 * L, 3 * L, 50)
    xgrid, ygrid = np.meshgrid(xvals, xvals)
    blob = rotated_blob(np.array([lx, ly, theta]), 0, 0, xgrid, ygrid)

    fig, ax = plt.subplots()
    ax.imshow(blob, interpolation="spline16", extent=[-3 * L, 3 * L, -3 * L, 3 * L])
    plt.savefig("blob.png", bbox_inches="tight")
    plt.show()

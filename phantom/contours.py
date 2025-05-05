# import numpy
import numpy as np

# Import package to compute level set
from skimage import measure

# import function to calculate area of closed curve
from shapely.geometry import Polygon, Point

# import function to compute convex hull of polygon
from scipy.spatial import ConvexHull

# Import package for progress bar
from tqdm.notebook import tqdm

# import library for parallel computing
from joblib import Parallel, delayed


def find_outermost_contour(X, Y, LAVD, distance, n, c_d, l_min, loc_threshold, Ncores):
    """
    The outermost nearly convex contours from the Lagrangian averaged Vorticity Deviation (LAVD)
    are extracted by specifying the maximum allowed convexity deficiendy "c_d" and minimum length "l_min"

    Parameters:
        X:              array (NY, NX)  X-meshgrid
        Y:              array (NY, NX)  Y-meshgrid
        LAVD:           array(Ny, Nx), LAVD-field
        n:              float, resolution of contour increments
        c_d:            float, convexity deficiency: generally in the interval [10^(-6), 10^{-3}]
        l_min:          float, minimum length of vortex boundary
        loc_threshold:  float, local threshold on LAVD to find local maxima
        Ncores:         int, number of cores to be used for parallel computing

    Returns:
        LAVD: list(N,), list of vortices
    """

    # compute local minimum of LAVD
    min_LAVD = np.nanmin(LAVD)  # float

    # find local maxima in LAVD
    idx_x, idx_y, loc_max_x, loc_max_y, loc_max_field = _loc_max(
        distance, X, Y, LAVD, loc_threshold
    )

    # define grid spacing
    dx = X[0, 1] - X[0, 0]  # float
    dy = Y[1, 0] - Y[0, 0]  # float

    # iterate over all local maxima and find outermost level set
    # of LAVD satisfying certain conditions listed above
    def parallel_iteration(i):

        # initialize vortex to np.nan
        B = [np.nan, np.nan]  # list (2, )

        # Break statement for loops
        BREAK = False

        # Point object: local maximum
        C = Point(X[idx_y[i], idx_x[i]], Y[idx_y[i], idx_x[i]])

        # iterate over level sets
        for j in np.linspace(min_LAVD, loc_max_field[i], n):

            # extract the x_0(\lambda,\phi_0)
            contour = measure.find_contours(LAVD, j)  # list

            # iterate over contours associated to level set j
            for c in contour:

                if c.shape[0] <= 4:
                    break

                # coordinates of contour
                x_polygon = np.min(X) + c[:, 1] * dx  # array
                y_polygon = np.min(Y) + c[:, 0] * dy  # array

                # create polygon object
                polygon = Polygon(np.array([x_polygon, y_polygon]).T)  # Polygon object

                # check if local maximum is inside contour and if polygon is closed
                if polygon.contains(C) and c[0, 1] == c[-1, 1] and c[0, 0] == c[-1, 0]:

                    # create convex hull
                    convex = ConvexHull(np.array([x_polygon, y_polygon]).T)

                    # Area of convex polygon
                    # (convex.volume returns the area, whereas convex.area returns the length of the perimeter in the two dimensional case)
                    A_convex = convex.volume  # float

                    # Area of polygon
                    A = polygon.area  # float

                    # Length of polygon
                    L = polygon.length  # float

                    # calculate convexity deficiency:
                    convexity_deficiency = abs((A_convex - A) / A_convex)  # float

                    # if condition is satisfied --> break inner for loop as the vortex boundary associated to the local maximum has been found.
                    if L > l_min and convexity_deficiency < c_d:
                        B = [x_polygon, y_polygon]
                        BREAK = True
                        break

            # Break out of second inner loop
            if BREAK:
                break

        return B

    # Compute vortex from LAVD with parallel computing
    vortex = Parallel(n_jobs=Ncores, verbose=0)(
        delayed(parallel_iteration)(i) for i in tqdm(range(len(loc_max_field)))
    )

    return vortex


def find_contour(event, refx, refy, t):
    ref_pixel = Point(data.R[refx, refy], data.Z[refx, refy])
    peak = data.isel(x=refx, y=refy, time=t)

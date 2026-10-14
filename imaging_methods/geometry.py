"""The last closed flux surface of an EFIT reconstruction, as a spline.

Moved from fusion_scripts `plotting_scripts/figure_plots.py` at 12b8543.
"""

import numpy as np
from scipy import interpolate

__all__ = ["calculate_splinted_LCFS"]


def calculate_splinted_LCFS(
    time_step: float,
    efit_time: np.ndarray,
    rbbbs: np.ndarray,
    zbbbs: np.ndarray,
):
    """The outboard leg of the last closed flux surface at the EFIT time nearest
    ``time_step``: ``(r_fine, z_fine)``.

    ``rbbbs`` and ``zbbbs`` are EFIT's boundary points, (point, efit_time), in
    centimetres, as the Alcator C-Mod files store them (``rlcfs``, ``zlcfs``), and
    ``efit_time`` their times. Of the slice nearest ``time_step``, the points with
    R >= 86 cm (the outboard leg) are interpolated by a cubic spline of R in Z, which is
    evaluated at ``z_fine``, 100 heights from Z = -8 cm to Z = 1 cm (the height of the
    APD view).

    The 86 cm and the -8 to 1 cm are C-Mod's, in centimetres, and carry to no other
    machine. ``scipy.interpolate.interp1d`` raises where the leg does not span those
    heights.
    """
    rbbbs = np.array(rbbbs)
    zbbbs = np.array(zbbbs)

    time_difference = np.absolute(time_step - efit_time)
    time_index = time_difference.argmin()

    closest_rbbbs = rbbbs[:, time_index]
    closest_zbbbs = zbbbs[:, time_index]

    f = interpolate.interp1d(
        closest_zbbbs[closest_rbbbs >= 86],
        closest_rbbbs[closest_rbbbs >= 86],
        kind="cubic",
    )
    z_fine = np.linspace(-8, 1, 100)
    r_fine = f(z_fine)

    return r_fine, z_fine

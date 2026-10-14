import netCDF4  # noqa: F401  -- before h5py, which velocity_estimation imports, or every netCDF read fails

from .method_parameters import (
    PreprocessingParams,
    TwoDcaParams,
    GaussFitParams,
    ContouringParams,
    TaudEstimationParams,
    PositionFilterParams,
    VelocityParams,
    MethodParameters,
)
from .utils import (
    PhantomDataInterface,
    MaskedImagingDataInterface,
    get_dt,
    get_dr,
    get_dz,
    autocorrelation,
    power_spectral_density,
    restrict_to_largest_true_subarray,
    smooth_da,
    get_default_synthetic_method_params,
    get_default_apd_method_params,
)
from .cond_av import find_events_and_2dca
from .contours import (
    compute_contour_mass,
    indexes_to_coordinates,
    fit_ellipses_to_contour_ds,
    get_contour_evolution,
    neighbour_level,
)
from .duration_time_estimation import (
    Analytics,
    SecondOrderStatistic,
    DurationTimeEstimator,
)
from .parameter_estimation import (
    rotated_blob,
    ellipse_parameters,
    fit_ellipse,
    fit_ellipse_to_event,
    estimate_fwhm_sizes,
)
from .data_preprocessing import interpolate_nans_3d, run_norm_ds
from .maximum_trajectory import (
    get_physical_pos,
    compute_maximum_trajectory_da,
    find_maximum_for_frame,
    parabolic_2d_interp,
    get_maximum_time,
)
from .synthetic_data import get_blob, make_2d_realization
from .velocity_estimates import (
    get_velocity_from_position,
    get_averaged_velocity_from_position,
    is_position_near_reference,
    get_combined_mask,
    get_averaged_velocity,
    get_positions_and_mask,
    estimate_velocities_synthetic_ds,
)
from .fitting import fit_two_sided, fit_exp
from .geometry import calculate_splinted_LCFS
from . import decorrelation, dead_pixels, puff

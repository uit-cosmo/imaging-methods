from dataclasses import dataclass, field


@dataclass
class PreprocessingParams:
    radius: int = 1000


@dataclass
class TwoDcaParams:
    """
    refx (int): X index of reference pixel
    refy (int): Y index of reference pixel
    threshold (float): Threshold value for event detection
    window (int): Size of window to extract around peaks
    check_max (int): Radius of the area on which the reference pixel is checked to be maximum at peak time. Set to 0 if
        no checking is desired.
    single_counting (bool): If True, ensures a minimum distance between events given by window_size.
    """

    refx: int = 8
    refy: int = 8
    threshold: float = 2.5
    window: int = 60
    check_max: int = 1
    single_counting: bool = True


@dataclass
class GaussFitParams:
    size_penalty: float = 5
    aspect_penalty: float = 0.2
    tilt_penalty: float = 0.2


@dataclass
class ContouringParams:
    threshold_factor: float = 0.5


@dataclass
class TaudEstimationParams:
    cutoff: float = 1e6
    nperseg: float = 1e3


@dataclass
class PositionFilterParams:
    window_size: int = 11
    window_type: str = "hann"
    mask_distance: float = 1
    mask_signal_factor: float = 0.75


@dataclass
class MethodParameters:
    """
    Dataclass containing setting for the applied method analyzing blob data.
        - preprocessing contains settings relevant for the preprocessing step
        - two_dca Two-dimensional conditional averaging settings
        - gauss_fitt: Setting for gaussian fitting
        - contouring: Contouring settings
        - taud_estimation: Settings for duration time estimation
        - position_filter: Length of the smoothing filter used for the position
    """

    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    two_dca: TwoDcaParams = field(default_factory=TwoDcaParams)
    gauss_fit: GaussFitParams = field(default_factory=GaussFitParams)
    contouring: ContouringParams = field(default_factory=ContouringParams)
    taud_estimation: TaudEstimationParams = field(default_factory=TaudEstimationParams)
    position_filter: PositionFilterParams = field(default_factory=PositionFilterParams)

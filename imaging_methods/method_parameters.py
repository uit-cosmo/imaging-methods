from dataclasses import dataclass, field


@dataclass
class PreprocessingParams:
    radius: int = 1000


@dataclass
class TwoDcaParams:
    refx: int = 8
    refy: int = 8
    threshold: int = 2
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
    com_smoothing: int = 11


@dataclass
class TaudEstimationParams:
    cutoff: float = 1e6
    nperseg: float = 1e3


@dataclass
class MethodParameters:
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    two_dca: TwoDcaParams = field(default_factory=TwoDcaParams)
    gauss_fit: GaussFitParams = field(default_factory=GaussFitParams)
    contouring: ContouringParams = field(default_factory=ContouringParams)
    taud_estimation: TaudEstimationParams = field(default_factory=TaudEstimationParams)

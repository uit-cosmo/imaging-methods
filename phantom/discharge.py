import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import os
from .data_preprocessing import load_data_and_preprocess
import numpy as np


@dataclass
class PlasmaDischarge:
    shot_number: int
    plasma_current: float  # Ip in MA
    line_averaged_density: float  # nebar in 10^20 m^-3
    greenwald_fraction: float  # fgw
    t_start: float  # in seconds
    t_end: float  # in seconds
    duration: float  # T in seconds
    mlp_mode: str  # Mirror Langmuir Probe mode
    confinement_mode: str

    def to_dict(self):
        """Convert the discharge data to a dictionary for JSON serialization."""
        return {
            "shot_number": self.shot_number,
            "plasma_current": self.plasma_current,
            "line_averaged_density": self.line_averaged_density,
            "greenwald_fraction": self.greenwald_fraction,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration": self.duration,
            "mlp_mode": self.mlp_mode,
            "confinement_more": self.confinement_mode,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a PlasmaDischarge instance from a dictionary."""
        return cls(
            shot_number=data["shot_number"],
            plasma_current=data["plasma_current"],
            line_averaged_density=data["line_averaged_density"],
            greenwald_fraction=data["greenwald_fraction"],
            t_start=data["t_start"],
            t_end=data["t_end"],
            duration=data["duration"],
            mlp_mode=data["mlp_mode"],
            confinement_mode=data["confinement_more"],
        )


class PlasmaDischargeManager:
    def __init__(self):
        self.discharges = []

    def add_discharge(self, discharge: PlasmaDischarge):
        """Add a plasma discharge to the manager."""
        self.discharges.append(discharge)

    def save_to_json(self, filename: str):
        """Save all discharges to a JSON file."""
        data = [discharge.to_dict() for discharge in self.discharges]
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, filename: str):
        """Load discharges from a JSON file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        with open(filename, "r") as f:
            data = json.load(f)
        self.discharges = [PlasmaDischarge.from_dict(item) for item in data]

    def get_discharge_by_shot(self, shot_number: int) -> Optional[PlasmaDischarge]:
        """Retrieve a discharge by its shot number."""
        for discharge in self.discharges:
            if discharge.shot_number == shot_number:
                return discharge
        return None

    def get_shot_list(self) -> List[int]:
        """Return a list of all shot numbers."""
        return [
            discharge.shot_number
            for discharge in self.discharges
            if discharge.confinement_mode == "L"
        ]

    def read_shot_data(
        self,
        shot,
        window=None,
        data_folder: str = "data",
        preprocessed=True,
        radius=1000,
    ):
        ds_full = load_data_and_preprocess(
            shot, window, data_folder, preprocessed, radius=radius
        )
        t_start, t_end = (
            self.get_discharge_by_shot(shot).t_start,
            self.get_discharge_by_shot(shot).t_end,
        )
        return ds_full.sel(time=slice(t_start, t_end))

    def get_data_from_tree(self, shot, data_folder="data"):
        try:
            print(f"Retrieving APD data for shot number: {shot}")
            import cmod_functions as cmod

            ds = cmod.generate_raw_apd_dataset(
                shot_number=shot, subtract_background=True
            )
            file_name = os.path.join(data_folder, f"apd_{shot}.nc")
            ds.to_netcdf(file_name)
        except Exception as e:
            print(f"Error retrieving APD data for shot {shot}: {str(e)}")


@dataclass
class BlobParameters:
    """
    Class describing the output blob parameters obtained with two-dimensional conditional averaging (2DCA)
    for GPI-APD data analysis. Stores velocity, area, shape, lifetime, and spatial scale parameters,
    with methods for validation, derived quantities, and data export.

    vx_c, vy_c and area_c are obtained with contouring methods.
    lx_f, ly_f, theta_f are obtained with a Gaussian fit.
    vx_tde, vy_tde are obtained with a 3-point TDE method.
    taud_psd and lambda_psd are obtained from fitting power spectral density.

    Attributes:
        vx_c (float): Center-of-mass velocity in x-direction (pixels/time step or m/s).
        vy_c (float): Center-of-mass velocity in y-direction (pixels/time step or m/s).
        area_c (float): Blob area (pixels² or m²).
        vx_2dca_tde (float): Velocity from TDE on 2DCA.
        vy_2dca_tde (float): Velocity from TDE on 2DCA.
        vx_tde (float): Velocity from TDE on GPI data.
        vy_tde (float): Velocity from TDE on GPI data..
        lx_f (float): Semi-major axis from Gaussian fit (pixels or m).
        ly_f (float): Semi-minor axis from Gaussian fit (pixels or m).
        lr (floar): FWHM radial size (m)
        lz (floar): FWHM radial size (m)
        theta_f (float): Ellipse rotation angle from Gaussian fit (radians).
        taud_psd (float): Blob lifetime from PSD analysis (time steps or ps).
        lambda_psd (float): Characteristic spatial scale from PSD analysis (pixels or m).
        number_events (int): Number of detected events.
    """

    vx_c: float
    vy_c: float
    area_c: float
    vx_2dca_tde: float
    vy_2dca_tde: float
    vx_tde: float
    vy_tde: float
    lx_f: float
    ly_f: float
    lr: float
    lz: float
    theta_f: float
    taud_psd: float
    lambda_psd: float
    number_events: float

    @property
    def velocity_c(self) -> Tuple[float, float]:
        """Return center-of-mass velocity vector (vx_c, vy_c)."""
        return (self.vx_c, self.vy_c)

    @property
    def velocity_tde(self) -> Tuple[float, float]:
        """Return time-dependent ellipse velocity vector (vx_tde, vy_tde)."""
        return (self.vx_2dca_tde, self.vy_2dca_tde)

    @property
    def total_velocity_c(self) -> float:
        """Compute magnitude of center-of-mass velocity."""
        return np.sqrt(self.vx_c**2 + self.vy_c**2)

    @property
    def total_velocity_tde(self) -> float:
        """Compute magnitude of time-dependent ellipse velocity."""
        return np.sqrt(self.vx_2dca_tde**2 + self.vy_2dca_tde**2)

    @property
    def aspect_ratio(self) -> float:
        """Compute aspect ratio of the fitted ellipse (lx_f / ly_f)."""
        return self.lx_f / self.ly_f if self.ly_f > 0 else np.inf

    @property
    def eccentricity(self) -> float:
        """Compute eccentricity of the fitted ellipse."""
        if self.lx_f >= self.ly_f:
            a, b = self.lx_f, self.ly_f
        else:
            a, b = self.ly_f, self.lx_f
        return np.sqrt(1 - (b / a) ** 2) if a > b else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to a dictionary for serialization."""
        return {
            "vx_c": self.vx_c,
            "vy_c": self.vy_c,
            "area_c": self.area_c,
            "vx_2dca_tde": self.vx_2dca_tde,
            "vy_2dca_tde": self.vy_2dca_tde,
            "vx_tde": self.vx_tde,
            "vy_tde": self.vy_tde,
            "lx_f": self.lx_f,
            "ly_f": self.ly_f,
            "lr": self.lr,
            "lz": self.lz,
            "theta_f": self.theta_f,
            "taud_psd": self.taud_psd,
            "lambda_psd": self.lambda_psd,
            "number_events": self.number_events,
        }

    def __str__(self) -> str:
        """String representation of the blob parameters."""
        return (
            f"BlobParameters(vx_c={self.vx_c:.2f}, vy_c={self.vy_c:.2f}, "
            f"area_c={self.area_c:.2f}, vx_2dca_tde={self.vx_2dca_tde:.2f}, vy_2dca_tde={self.vy_2dca_tde:.2f},"
            f" vx_tde={self.vx_tde:.2f}, vy_tde={self.vy_tde:.2f}"
            f"lx_f={self.lx_f:.2f}, ly_f={self.ly_f:.2f}, lr={self.lr:.2f}, lz={self.lz:.2f}, theta_f={self.theta_f:.2f}, "
            f"taud_psd={self.taud_psd:.2f}, lambda_psd={self.lambda_psd:.2f}, number_events={self.number_events})"
        )


@dataclass
class ShotData:
    """
    A class to encapsulate a PlasmaDischarge and BlobParameters for a single shot.

    Attributes:
    -----------
    discharge : PlasmaDischarge
        Experimental parameters for the plasma discharge.
    blob_params : BlobParameters
        Blob parameters obtained from analysis.

    Methods:
    --------
    to_dict():
        Convert the shot data to a dictionary for JSON serialization.
    from_dict(cls, data):
        Create a ShotData instance from a dictionary.
    """

    discharge: PlasmaDischarge
    blob_params: Dict[int, Dict[int, Optional[BlobParameters]]]

    def to_dict(self):
        """Convert the shot data to a dictionary for JSON serialization."""
        return {
            "plasma_discharge": self.discharge.to_dict(),
            "blob_params": {
                str(refx): {
                    str(refy): params.to_dict() if params is not None else None
                    for refy, params in refy_dict.items()
                }
                for refx, refy_dict in self.blob_params.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Union[Dict, Dict[str, Dict[str, Optional[Dict]]]]]
    ):
        """Create a ShotData instance from a dictionary."""
        required_keys = {"plasma_discharge", "blob_params"}
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            raise KeyError(f"Missing required keys in JSON data: {missing}")
        discharge = PlasmaDischarge.from_dict(data["plasma_discharge"])
        blob_params = {
            int(refx): {
                int(refy): BlobParameters(**params) if params is not None else None
                for refy, params in refy_dict.items()
            }
            for refx, refy_dict in data["blob_params"].items()
        }
        return cls(discharge=discharge, blob_params=blob_params)


class ScanResults:
    """
    A class to manage a collection of ShotData instances for multiple shots.

    Attributes:
    -----------
    shots : List[ShotData]
        List of ShotData instances, each containing a PlasmaDischarge and BlobParameters.

    Methods:
    --------
    add_shot(discharge, blob_params):
        Add a ShotData instance to the collection.
    print_summary():
        Print a summary of all shots in the collection.
    to_json(filename=None):
        Save the collection to a JSON file or return as a JSON string.
    from_json(cls, json_data=None, filename=None):
        Create a ScanResults instance from a JSON file or string.
    """

    def __init__(self, shots: Optional[Dict[int, ShotData]] = None):
        """Initialize a ScanResults instance."""
        self.shots = {} if shots is None else shots
        if shots is not None:
            for shot_number, shot_data in shots.items():
                if not isinstance(shot_number, int):
                    raise TypeError("Shot numbers must be integers")
                if not isinstance(shot_data, ShotData):
                    raise TypeError("All values in shots must be ShotData instances")

    def add_shot(self, discharge: PlasmaDischarge, blob_params: BlobParameters):
        """
        Add a ShotData instance to the collection.

        Parameters:
        -----------
        discharge : PlasmaDischarge
            The PlasmaDischarge instance to add.
        blob_params : BlobParameters
            The BlobParameters instance to add.

        Raises:
        -------
        TypeError
            If discharge is not a PlasmaDischarge instance or blob_params is not a BlobParameters instance.
        """
        shot_data = ShotData(discharge=discharge, blob_params=blob_params)
        self.shots[discharge.shot_number] = shot_data

    def add_blob_params(
        self, shot_number: int, refx: int, refy: int, blob_params: BlobParameters
    ):
        if not isinstance(blob_params, BlobParameters):
            raise TypeError("blob_params must be a BlobParameters instance")
        if not isinstance(refx, int) or not isinstance(refy, int):
            raise TypeError("refx and refy must be integers")

        # Check if shot exists
        if shot_number not in self.shots:
            # Requires discharge data to create a new ShotData instance
            raise ValueError(
                f"Shot {shot_number} does not exist. Add discharge data first using add_shot."
            )

        # Get the ShotData instance
        shot_data = self.shots[shot_number]

        # Initialize refx dictionary if it doesn't exist
        if refx not in shot_data.blob_params:
            shot_data.blob_params[refx] = {}

        # Add or update blob_params for the given refx, refy
        shot_data.blob_params[refx][refy] = blob_params

    def to_json(self, filename):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
                return super().default(obj)

        data = {str(k): v.to_dict() for k, v in self.shots.items()}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=CustomEncoder, indent=4)

    @classmethod
    def from_json(cls, filename: str):
        def object_hook(data):
            if (
                isinstance(data, dict)
                and "plasma_discharge" in data
                and "blob_params" in data
            ):
                return ShotData.from_dict(data)
            return data

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f, object_hook=object_hook)

        shots = {int(k): v for k, v in data.items()}
        return cls(shots=shots)


class ResultsManager:
    def __init__(self, results_file: str = "results.json"):
        self.results_file = results_file
        self.results = None
        self.load_data()

    def load_data(self) -> None:
        if os.path.exists(self.results_file):
            self.results = ScanResults.from_json(filename=self.results_file)
        else:
            self.results = ScanResults()

    def get_results(self, shot_number: int, refx: int, refy: int) -> Optional[ShotData]:
        if self.results is None or shot_number not in self.results.shots:
            return None
        shot_data = self.results.shots[shot_number]
        if refx not in shot_data.blob_params or refy not in shot_data.blob_params[refx]:
            return None
        return shot_data if shot_data.blob_params[refx][refy] is not None else None

    def get_lr(
        self, shot_number: int, refx: Union[int, List[int]], refy: Union[int, List[int]]
    ) -> float:
        return self.get_blob_param(shot_number, refx, refy, "lr")

    def get_lz(
        self, shot_number: int, refx: Union[int, List[int]], refy: Union[int, List[int]]
    ) -> float:
        return self.get_blob_param(shot_number, refx, refy, "lz")

    def get_blob_param(
        self,
        shot_number: int,
        refx: Union[int, List[int]],
        refy: Union[int, List[int]],
        param_name: str,
    ) -> float:
        """Retrieve a blob_params attribute value or average over refx and refy if provided as lists. Returns np.nan if no valid data."""
        # Convert refx and refy to lists if they are integers
        refx_values = [refx] if isinstance(refx, int) else refx
        refy_values = [refy] if isinstance(refy, int) else refy

        # Collect parameter values for all combinations of refx and refy
        param_values = []
        for rx in refx_values:
            for ry in refy_values:
                result = self.get_results(shot_number, rx, ry)
                param = (
                    getattr(result.blob_params, param_name)
                    if result is not None
                    else np.nan
                )
                param_values.append(param)

        # Compute average, ignoring NaN values
        param_array = np.array(param_values)
        if np.all(np.isnan(param_array)):
            return np.nan
        return np.nanmean(param_array)

    def save_results(self):
        if self.results is not None:
            self.results.to_json(self.results_file)

    def add_shot_data(
        self,
        discharge: PlasmaDischarge,
        blob_params: Dict[int, Dict[int, Optional[BlobParameters]]],
    ):
        self.results.add_shot(discharge, blob_params)
        self.save_results()

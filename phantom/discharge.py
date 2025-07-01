import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
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
        return [discharge.shot_number for discharge in self.discharges]

    def read_shot_data(
        self,
        shot,
        window=None,
        data_folder: str = "data",
        preprocessed=True,
        radius=1000,
    ):
        return load_data_and_preprocess(
            shot, window, data_folder, preprocessed, radius=radius
        )


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
        vx_tde (float): Velocity from time-dependent ellipse fitting in x-direction.
        vy_tde (float): Velocity from time-dependent ellipse fitting in y-direction.
        lx_f (float): Semi-major axis from Gaussian fit (pixels or m).
        ly_f (float): Semi-minor axis from Gaussian fit (pixels or m).
        theta_f (float): Ellipse rotation angle from Gaussian fit (radians).
        taud_psd (float): Blob lifetime from PSD analysis (time steps or ps).
        lambda_psd (float): Characteristic spatial scale from PSD analysis (pixels or m).
        number_events (int): Number of detected events.
    """

    vx_c: float
    vy_c: float
    area_c: float
    vx_tde: float
    vy_tde: float
    lx_f: float
    ly_f: float
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
        return (self.vx_tde, self.vy_tde)

    @property
    def total_velocity_c(self) -> float:
        """Compute magnitude of center-of-mass velocity."""
        return np.sqrt(self.vx_c**2 + self.vy_c**2)

    @property
    def total_velocity_tde(self) -> float:
        """Compute magnitude of time-dependent ellipse velocity."""
        return np.sqrt(self.vx_tde**2 + self.vy_tde**2)

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
            "vx_tde": self.vx_tde,
            "vy_tde": self.vy_tde,
            "lx_f": self.lx_f,
            "ly_f": self.ly_f,
            "theta_f": self.theta_f,
            "taud_psd": self.taud_psd,
            "lambda_psd": self.lambda_psd,
            "number_events": self.number_events,
        }

    def __str__(self) -> str:
        """String representation of the blob parameters."""
        return (
            f"BlobParameters(vx_c={self.vx_c:.2f}, vy_c={self.vy_c:.2f}, "
            f"area_c={self.area_c:.2f}, vx_tde={self.vx_tde:.2f}, vy_tde={self.vy_tde:.2f}, "
            f"lx_f={self.lx_f:.2f}, ly_f={self.ly_f:.2f}, theta_f={self.theta_f:.2f}, "
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
    blob_params: BlobParameters

    def __post_init__(self):
        """Validate the types of discharge and blob_params."""
        if not isinstance(self.discharge, PlasmaDischarge):
            raise TypeError("discharge must be a PlasmaDischarge instance")
        if not isinstance(self.blob_params, BlobParameters):
            raise TypeError("blob_params must be a BlobParameters instance")

    def to_dict(self) -> Dict[str, Dict]:
        """Convert the shot data to a dictionary for JSON serialization."""
        return {
            "plasma_discharge": self.discharge.to_dict(),
            "blob_parameters": self.blob_params.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "ShotData":
        """Create a ShotData instance from a dictionary."""
        required_keys = {"plasma_discharge", "blob_parameters"}
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            raise KeyError(f"Missing required keys in JSON data: {missing}")
        discharge = PlasmaDischarge.from_dict(data["plasma_discharge"])
        blob_params = BlobParameters(**data["blob_parameters"])
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

    def __init__(self, shots: Optional[List[ShotData]] = None):
        """
        Initialize a ScanResults instance.

        Parameters:
        -----------
        shots : List[ShotData], optional
            Initial list of ShotData instances. If None, starts with an empty list.

        Raises:
        -------
        TypeError
            If any element in shots is not a ShotData instance.
        """
        self.shots = []
        if shots is not None:
            for shot in shots:
                if not isinstance(shot, ShotData):
                    raise TypeError("All elements in shots must be ShotData instances")
                self.shots.append(shot)

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
        self.shots.append(shot_data)

    def print_summary(self):
        """
        Print a summary of all shots in the collection.
        """
        if not self.shots:
            print("No shots in the collection.")
            return
        print(f"Scan Results Summary ({len(self.shots)} shots):")
        for i, shot in enumerate(self.shots, 1):
            print(f"\nShot {i}:")
            print("  Plasma Discharge Parameters:")
            print(f"    Shot Number: {shot.discharge.shot_number}")
            print(f"    Plasma Current: {shot.discharge.plasma_current:.2f} (MA)")
            print(
                f"    Line-Averaged Density: {shot.discharge.line_averaged_density:.2f} (10^20 m^-3)"
            )
            print(f"    Greenwald Fraction: {shot.discharge.greenwald_fraction:.2f}")
            print(f"    Start Time: {shot.discharge.t_start:.2f} (s)")
            print(f"    End Time: {shot.discharge.t_end:.2f} (s)")
            print(f"    Duration: {shot.discharge.duration:.2f} (s)")
            print(f"    MLP Mode: {shot.discharge.mlp_mode}")
            print(f"    Confinement Mode: {shot.discharge.confinement_mode}")
            print("  Blob Parameters:")
            print(f"    {shot.blob_params}")

    def to_json(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save the collection to a JSON file or return as a JSON string.

        Parameters:
        -----------
        filename : str, optional
            If provided, save the results to this file. If None, return a JSON string.

        Returns:
        --------
        str or None
            JSON string if filename is None, otherwise None.

        Raises:
        -------
        OSError
            If writing to the file fails.
        """
        data = [shot.to_dict() for shot in self.shots]
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            return None
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, json_data: Optional[str] = None, filename: Optional[str] = None):
        """
        Create a ScanResults instance from a JSON file or string.

        Parameters:
        -----------
        json_data : str, optional
            JSON string containing the collection data. Ignored if filename is provided.
        filename : str, optional
            Path to a JSON file containing the collection data.

        Returns:
        --------
        ScanResults
            A new ScanResults instance with the loaded shots.

        Raises:
        -------
        ValueError
            If neither json_data nor filename is provided, or if JSON data is invalid.
        OSError
            If reading the file fails.
        KeyError
            If required keys are missing from the JSON data for any shot.
        """
        if filename:
            with open(filename, "r") as f:
                data = json.load(f)
        elif json_data:
            data = json.loads(json_data)
        else:
            raise ValueError("Either json_data or filename must be provided")

        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of shot dictionaries")

        shots = [ShotData.from_dict(item) for item in data]
        return cls(shots=shots)

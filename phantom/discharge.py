import json
from dataclasses import dataclass
from typing import Optional, List
import os
from .utils import get_sample_data


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
        self, shot, window=None, data_folder: str = "data", preprocessed=True
    ):
        return get_sample_data(shot, window, data_folder, preprocessed)


class ShotAnalysis:
    """
    A class to store and manage analysis results for a single shot.

    Attributes:
    -----------
    shot : int
        Shot number
    v : float
        Radial velocity component (e.g., in m/s).
    w : float
        Poloidal velocity component (e.g., in m/s).
    lx : float
        Length scale or coordinate in x-direction (e.g., in m).
    ly : float
        Length scale or coordinate in y-direction (e.g., in m).
    theta : float
        Angle (e.g., in radians).
    taud : float
        Duration time parameter from PSD fitting (in seconds).
    lam : float
        Asymmetry parameter from PSD fitting, in [0, 1].

    Methods:
    --------
    print_results():
        Print the analysis results in a formatted manner.
    """

    def __init__(self, shot, v, w, lx, ly, theta, taud, lam):
        """
        Initialize a ShotAnalysis instance with the given parameters.

        Parameters:
        -----------
        shot, v, w, lx, ly, theta, taud, lam : float
            Values for the respective attributes. taud must be positive, and lam must be in [0, 1].

        Raises:
        -------
        ValueError
            If taud <= 0 or lam is not in [0, 1].
        TypeError
            If any parameter is not a number.
        """
        # Validate numeric inputs
        for param, name in [
            (v, "v"),
            (w, "w"),
            (lx, "lx"),
            (ly, "ly"),
            (theta, "theta"),
            (taud, "taud"),
            (lam, "lam"),
        ]:
            if not isinstance(param, (int, float, np.number)):
                raise TypeError(f"{name} must be a number")

        # Validate physical constraints
        if taud <= 0:
            raise ValueError("taud must be positive")
        if not 0 <= lam <= 1:
            raise ValueError("lam must be in [0, 1]")

        self.shot = shot
        self.v = float(v)
        self.w = float(w)
        self.lx = float(lx)
        self.ly = float(ly)
        self.theta = float(theta)
        self.taud = float(taud)
        self.lam = float(lam)

    def print_results(self):
        """
        Print the analysis results in a formatted manner.
        """
        print("Shot Analysis Results:")
        print(f"  shot: {self.shot}")
        print(f"  v: {self.v:.2f} (m/s or Hz)")
        print(f"  w: {self.w:.2f} (m/s or Hz)")
        print(f"  lx: {self.lx:.2f} (m)")
        print(f"  ly: {self.ly:.2f} (m)")
        print(f"  theta: {self.theta:.2f} (rad)")
        print(f"  taud: {self.taud:.2g} (s)")
        print(f"  lambda: {self.lam:.2g} (dimensionless)")

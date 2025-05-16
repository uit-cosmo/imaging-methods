import json
from dataclasses import dataclass
from typing import Optional, List
import os
from .utils import get_sample_data
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
        self, shot, window=None, data_folder: str = "data", preprocessed=True
    ):
        return get_sample_data(shot, window, data_folder, preprocessed)


class ShotAnalysis:
    """
    A class to store and manage analysis results for a single shot.

    Attributes:
    -----------
    v : float
        Radial velocity component (in m/s).
    w : float
        Poloidal velocity component (in m/s).
    lx : float
        Length scale or coordinate in x-direction (in meters).
    ly : float
        Length scale or coordinate in y-direction (in meters).
    theta : float
        Angle (e.g., in radians).
    taud : float
        Decay time parameter from PSD fitting (in seconds).
    lam : float
        Weighting factor from PSD fitting, in [0, 1].
    plasma_discharge : PlasmaDischarge
        Experimental parameters for the plasma discharge.

    Methods:
    --------
    print_results():
        Print the analysis results in a formatted manner.
    to_json(filename=None):
        Save the analysis results to a JSON file or return as a JSON string.
    from_json(cls, json_data, filename=None):
        Create a ShotAnalysis instance from a JSON file or string.
    """

    def __init__(self, v, w, lx, ly, theta, taud, lam, plasma_discharge):
        """
        Initialize a ShotAnalysis instance with the given parameters.

        Parameters:
        -----------
        v, w, lx, ly, theta, taud, lam : float
            Values for the respective attributes. taud must be positive, and lam must be in [0, 1].
        plasma_discharge : PlasmaDischarge
            Experimental parameters for the plasma discharge.

        Raises:
        -------
        ValueError
            If taud <= 0 or lam is not in [0, 1].
        TypeError
            If any parameter is not a number or plasma_discharge is not a PlasmaDischarge instance.
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
        if not isinstance(plasma_discharge, PlasmaDischarge):
            raise TypeError("plasma_discharge must be a PlasmaDischarge instance")

        self.v = float(v)
        self.w = float(w)
        self.lx = float(lx)
        self.ly = float(ly)
        self.theta = float(theta)
        self.taud = float(taud)
        self.lam = float(lam)
        self.plasma_discharge = plasma_discharge

    def print_results(self):
        """
        Print the analysis results in a formatted manner.
        """
        print("Shot Analysis Results:")
        print(f"  v: {self.v:.2f} (m/s)")
        print(f"  w: {self.w:.2f} (m/s)")
        print(f"  lx: {self.lx:.2f} (m)")
        print(f"  ly: {self.ly:.2f} (m)")
        print(f"  theta: {self.theta:.2f} (rad)")
        print(f"  taud: {self.taud:.2g} (s)")
        print(f"  lambda: {self.lam:.2g} (dimensionless)")
        print("  Plasma Discharge Parameters:")
        print(f"    Shot Number: {self.plasma_discharge.shot_number}")
        print(f"    Plasma Current: {self.plasma_discharge.plasma_current:.2f} (MA)")
        print(
            f"    Line-Averaged Density: {self.plasma_discharge.line_averaged_density:.2f} (10^20 m^-3)"
        )
        print(f"    Greenwald Fraction: {self.plasma_discharge.greenwald_fraction:.2f}")
        print(f"    Start Time: {self.plasma_discharge.t_start:.2f} (s)")
        print(f"    End Time: {self.plasma_discharge.t_end:.2f} (s)")
        print(f"    Duration: {self.plasma_discharge.duration:.2f} (s)")
        print(f"    MLP Mode: {self.plasma_discharge.mlp_mode}")

    def to_json(self, filename=None):
        """
        Save the analysis results to a JSON file or return as a JSON string.

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
        data = {
            "v": self.v,
            "w": self.w,
            "lx": self.lx,
            "ly": self.ly,
            "theta": self.theta,
            "taud": self.taud,
            "lam": self.lam,
            "plasma_discharge": self.plasma_discharge.to_dict(),
        }
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        else:
            return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, json_data=None, filename=None):
        """
        Create a ShotAnalysis instance from a JSON file or string.

        Parameters:
        -----------
        json_data : str, optional
            JSON string containing the analysis results. Ignored if filename is provided.
        filename : str, optional
            Path to a JSON file containing the analysis results.

        Returns:
        --------
        ShotAnalysis
            A new ShotAnalysis instance with the loaded parameters.

        Raises:
        -------
        ValueError
            If neither json_data nor filename is provided, or if JSON data is invalid.
        OSError
            If reading the file fails.
        KeyError
            If required keys are missing from the JSON data.
        """
        if filename:
            with open(filename, "r") as f:
                data = json.load(f)
        elif json_data:
            data = json.loads(json_data)
        else:
            raise ValueError("Either json_data or filename must be provided")

        # Validate required keys
        required_keys = {
            "v",
            "w",
            "lx",
            "ly",
            "theta",
            "taud",
            "lam",
            "plasma_discharge",
        }
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            raise KeyError(f"Missing required keys in JSON data: {missing}")

        plasma_discharge = PlasmaDischarge.from_dict(data["plasma_discharge"])
        return cls(
            v=data["v"],
            w=data["w"],
            lx=data["lx"],
            ly=data["ly"],
            theta=data["theta"],
            taud=data["taud"],
            lam=data["lam"],
            plasma_discharge=plasma_discharge,
        )


class ScanResults:
    """
    A class to manage a collection of ShotAnalysis instances for multiple shots.

    Attributes:
    -----------
    shots : list
        List of ShotAnalysis instances representing individual shots.

    Methods:
    --------
    add_shot(shot):
        Add a ShotAnalysis instance to the collection.
    print_summary():
        Print a summary of all shots in the collection.
    to_json(filename=None):
        Save the collection to a JSON file or return as a JSON string.
    from_json(cls, json_data=None, filename=None):
        Create a ShotCollection instance from a JSON file or string.
    """

    def __init__(self, shots=None):
        """
        Initialize a ShotCollection instance.

        Parameters:
        -----------
        shots : list, optional
            Initial list of ShotAnalysis instances. If None, starts with an empty list.

        Raises:
        -------
        TypeError
            If any element in shots is not a ShotAnalysis instance.
        """
        self.shots = []
        if shots is not None:
            for shot in shots:
                if not isinstance(shot, ShotAnalysis):
                    raise TypeError(
                        "All elements in shots must be ShotAnalysis instances"
                    )
                self.shots.append(shot)

    def add_shot(self, shot):
        """
        Add a ShotAnalysis instance to the collection.

        Parameters:
        -----------
        shot : ShotAnalysis
            The ShotAnalysis instance to add.

        Raises:
        -------
        TypeError
            If shot is not a ShotAnalysis instance.
        """
        if not isinstance(shot, ShotAnalysis):
            raise TypeError("shot must be a ShotAnalysis instance")
        self.shots.append(shot)

    def print_summary(self):
        """
        Print a summary of all shots in the collection.
        """
        if not self.shots:
            print("No shots in the collection.")
            return
        print(f"Shot Collection Summary ({len(self.shots)} shots):")
        for i, shot in enumerate(self.shots, 1):
            print(f"\nShot {i}:")
            shot.print_results()

    def to_json(self, filename=None):
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
        data = [json.loads(shot.to_json()) for shot in self.shots]
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        else:
            return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, filename):
        """
        Create a ShotCollection instance from a JSON file or string.

        Parameters:
        -----------
        filename : str
            Path to a JSON file containing the collection data.

        Returns:
        --------
        ShotCollection
            A new ShotCollection instance with the loaded shots.

        Raises:
        -------
        OSError
            If reading the file fails.
        KeyError
            If required keys are missing from the JSON data for any shot.
        """
        if filename:
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            raise ValueError("Either json_data or filename must be provided")

        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of shot dictionaries")

        shots = [ShotAnalysis.from_json(json_data=json.dumps(shot)) for shot in data]
        return cls(shots=shots)

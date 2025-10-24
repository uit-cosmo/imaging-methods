import json
from dataclasses import dataclass
from typing import Optional, List
import os

import numpy as np


@dataclass
class PlasmaDischarge:
    shot_number: int
    plasma_current: float  # Ip in MA
    line_averaged_density: float  # nebar in 10^20 m^-3
    greenwald_fraction: float  # fgw
    t_start: float  # in seconds
    t_end: float  # in seconds
    mlp_mode: str  # Mirror Langmuir Probe mode
    comment: str

    def to_dict(self):
        """Convert the discharge data to a dictionary for JSON serialization."""
        return {
            "shot_number": self.shot_number,
            "plasma_current": self.plasma_current,
            "line_averaged_density": self.line_averaged_density,
            "greenwald_fraction": self.greenwald_fraction,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "mlp_mode": self.mlp_mode,
            "comment": self.comment,
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
            mlp_mode=data["mlp_mode"],
            comment=data["comment"],
        )

    def greenwald_fraction_fun(self):
        return self.line_averaged_density * np.pi * 0.22**2 / self.plasma_current


class PlasmaDischargeManager:
    def __init__(self, file=None):
        """Creates a PlasmaDischargeManager object, a json file can be specified to load the discharges"""
        self.discharges = []
        if file is not None:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found.")
            with open(file, "r") as f:
                data = json.load(f)
            self.discharges = [PlasmaDischarge.from_dict(item) for item in data]
            self.discharges.sort(key=lambda x: x.shot_number)

    def add_discharge(self, discharge: PlasmaDischarge):
        """Add a plasma discharge to the manager."""
        self.discharges.append(discharge)
        self.discharges.sort(key=lambda x: x.shot_number)

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
        self.discharges.sort(key=lambda x: x.shot_number)

    def get_discharge_by_shot(self, shot_number: int) -> Optional[PlasmaDischarge]:
        """Retrieve a discharge by its shot number."""
        for discharge in self.discharges:
            if discharge.shot_number == shot_number:
                return discharge
        return None

    def get_shot_list(self) -> List[int]:
        """Return a list of all shot numbers."""
        return [discharge.shot_number for discharge in self.discharges]

    def get_shot_list_by_confinement(self, confinement) -> List[int]:
        return [
            discharge.shot_number
            for discharge in self.discharges
            if discharge.comment in confinement
        ]

    def get_ohmic_H_shot_list(self) -> List[int]:
        return [
            1120814016,
            1120814018,
            1120814019,
            1120814020,
            1120814021,
            1120814025,
            1120814026,
            1120814027,
            1120814028,
            1120814029,
            1120814030,
            1120814031,
            1120814032,
        ]

    def get_ohmic_shot_list(self) -> List[int]:
        return [
            discharge.shot_number
            for discharge in self.discharges
            if discharge.comment == "L"
        ]

    def get_imode_shot_list(self) -> List[int]:
        return [
            discharge.shot_number
            for discharge in self.discharges
            if discharge.comment == "I-mode"
        ]

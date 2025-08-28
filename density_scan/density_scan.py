from phantom import PlasmaDischargeManager, PlasmaDischarge
import numpy as np

manager = PlasmaDischargeManager()
# Data from the provided table
discharges = [
    PlasmaDischarge(1160616027, 0.51, 2.82, 0.80, 1.15, 1.45, 0.300, "-", "L"),
    PlasmaDischarge(1160616026, 0.51, 2.83, 0.80, 1.15, 1.45, 0.300, "-", "L"),
    PlasmaDischarge(1160616025, 0.51, 2.76, 0.80, 1.15, 1.45, 0.300, "-", "L"),
    PlasmaDischarge(1160616022, 0.52, 2.29, 0.65, 1.15, 1.45, 0.300, "scan", "L"),
    PlasmaDischarge(1160616018, 0.53, 1.65, 0.46, 1.15, 1.45, 0.300, "dwell", "L"),
    PlasmaDischarge(1160616017, 0.53, 1.56, 0.44, 1.15, 1.45, 0.300, "scan", "L"),
    PlasmaDischarge(1160616016, 0.53, 1.60, 0.45, 1.15, 1.45, 0.300, "scan", "L"),
    PlasmaDischarge(1160616011, 0.54, 1.07, 0.30, 1.15, 1.45, 0.300, "scan", "L"),
    PlasmaDischarge(1160616009, 0.54, 0.87, 0.24, 1.305, 1.45, 0.145, "scan", "L"),
    PlasmaDischarge(1110201016, 0.90, 3.69, 0.64, 1.06, 1.345, 0.285, "none", "EDA-H"),
    PlasmaDischarge(
        1110201011, 1.20, 4.32, 0.56, 1.06, 1.255, 0.195, "none", "ELM-free-H"
    ),
    PlasmaDischarge(1120814016, 0.743, 1.635, 0.33, 1.3, 1.65, 0.35, "none", "EDA-H"),
    PlasmaDischarge(1120814018, 0.741, 1.621, 0.33, 1.33, 1.65, 0.32, "none", "EDA-H"),
    PlasmaDischarge(1120814019, 0.744, 1.601, 0.32, 1.33, 1.65, 0.32, "none", "EDA-H"),
    PlasmaDischarge(1120814020, 0.745, 1.706, 0.34, 1.33, 1.65, 0.32, "none", "EDA-H"),
    PlasmaDischarge(1120814021, 0.744, 1.987, 0.40, 1.33, 1.65, 0.32, "none", "EDA-H"),
    PlasmaDischarge(1120814025, 0.745, 1.572, 0.32, 1.33, 1.65, 0.32, "none", "EDA-H"),
    PlasmaDischarge(1120814026, 0.745, 1.278, 0.26, 0.95, 1.4, 0.45, "scan", "EDA-H"),
    PlasmaDischarge(1120814027, 0.743, 1.561, 0.32, 0.95, 1.4, 0.45, "scan", "EDA-H"),
    PlasmaDischarge(1120814028, 0.74, 1.77, 0.36, 1.15, 1.5, 0.35, "scan", "EDA-H"),
    PlasmaDischarge(1120814029, 0.746, 1.75, 0.35, 1.15, 1.5, 0.35, "scan", "EDA-H"),
    PlasmaDischarge(1120814030, 0.744, 1.74, 0.35, 1.15, 1.5, 0.35, "scan", "EDA-H"),
    PlasmaDischarge(1120814031, 0.743, 1.652, 0.33, 1.15, 1.5, 0.35, "scan", "EDA-H"),
    PlasmaDischarge(1120814032, 0.747, 1.83, 0.37, 1.15, 1.5, 0.35, "scan", "EDA-H"),
    PlasmaDischarge(
        1110201007, 0.93, np.nan, np.nan, 1.10, 1.40, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201010, 0.93, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201011, 1.22, np.nan, np.nan, 1.12, 1.25, np.nan, "none", "ELM-free-H"
    ),
    PlasmaDischarge(
        1110201014, 0.93, np.nan, np.nan, 1.05, 1.31, np.nan, "none", "ELM-free-H"
    ),
    PlasmaDischarge(
        1110201015, 0.91, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201016, 0.92, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201018, 0.93, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201021, 0.93, np.nan, np.nan, 1.23, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201022, 0.73, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201023, 0.74, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201025, 0.73, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201026, 0.73, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201027, 0.73, np.nan, np.nan, 1.05, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1110201034, 0.51, np.nan, np.nan, 1.11, 1.35, np.nan, "none", "EDA-H"
    ),
    PlasmaDischarge(
        1140613026, np.nan, np.nan, np.nan, 0.7, 1.2, np.nan, "none", "I-mode"
    ),
    PlasmaDischarge(
        1140613027, np.nan, np.nan, np.nan, 0.7, 1.2, np.nan, "none", "I-mode"
    ),
]
# Add discharges to manager
for discharge in discharges:
    manager.add_discharge(discharge)
# Save to JSON file
manager.save_to_json("plasma_discharges.json")
# Load from JSON file and print a sample
manager.load_from_json("plasma_discharges.json")
sample_discharge = manager.get_discharge_by_shot(1160616026)
if sample_discharge:
    print(f"Sample discharge: {sample_discharge}")

from phantom import PlasmaDischargeManager, PlasmaDischarge

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

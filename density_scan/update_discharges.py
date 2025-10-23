from experimental_database import *

discharge_manager = PlasmaDischargeManager("../../experimental_database/plasma_discharges.json")


print(len(discharge_manager.discharges))

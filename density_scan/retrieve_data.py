import phantom as ph

manager = ph.PlasmaDischargeManager()
manager.load_from_json("density_scan/plasma_discharges.json")

for shot in manager.get_shot_list():
    manager.get_data_from_tree(shot, "data")

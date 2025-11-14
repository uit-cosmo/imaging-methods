method_parameters = {
    "preprocessing": {"radius": 1000},
    "2dca": {
        "refx": 0,
        "refy": 3,
        "threshold": 2,
        "window": 60,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 3},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 2e3},
}

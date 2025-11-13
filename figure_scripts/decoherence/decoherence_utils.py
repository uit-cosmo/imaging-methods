import numpy as np

import imaging_methods as im
import matplotlib.pyplot as plt
from blobmodel import BlobShapeEnum, BlobShapeImpl
import velocity_estimation as ve
import os
import cosmoplots as cp
# src/main.py
import sys
from pathlib import Path

# Add the repository root (one level up) to the import search path
repo_root = Path(__file__).resolve().parent.parent   # src → my_project
sys.path.append(str(repo_root))

# Now you can import as if utils were in the same directory
from utils import *


# Method parameters
method_parameters = {
    "preprocessing": {"radius": 1000},
    "2dca": {
        "refx": 4,
        "refy": 4,
        "threshold": 2,
        "window": 30,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 10},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}


def estimate_velocities(ds, method_parameters):
    """
    Does a full analysis on imaging data, estimating a BlobParameters object containing all estimates. Plots figures
    when relevant in the provided figures_dir.
    """
    dt = im.get_dt(ds)

    tdca_params = method_parameters["2dca"]
    refx, refy = tdca_params["refx"], tdca_params["refy"]
    events, average_ds = im.find_events_and_2dca(
        ds,
        refx,
        refy,
        threshold=tdca_params["threshold"],
        check_max=tdca_params["check_max"],
        window_size=tdca_params["window"],
        single_counting=tdca_params["single_counting"],
    )

    contour_ds = im.get_contour_evolution(
        average_ds.cond_av,
        method_parameters["contouring"]["threshold_factor"],
        max_displacement_threshold=None,
    )

    velocity_ds = im.get_contour_velocity(
        contour_ds.center_of_mass,
        method_parameters["contouring"]["com_smoothing"],
    )

    v_c, w_c = velocity_ds.sel(time=slice(-3, 3)).mean(dim="time", skipna=True).values

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters["2dca"]["window"] * dt
    eo.cc_options.minimum_cc_value = 0
    pd = ve.estimate_velocities_for_pixel(
        refx, refy, ve.CModImagingDataInterface(ds)
    )
    vx_cc_tde, vy_cc_tde = pd.vx, pd.vy

    vx_2dca_tde, vy_2dca_tde = im.get_3tde_velocities(average_ds.cond_av, refx, refy)
    cond_repr_right = np.max(average_ds.cond_repr.isel(x=refx+1, y=refy).values)
    cond_repr_left = np.max(average_ds.cond_repr.isel(x=refx-1, y=refy).values)
    cond_repr_up = np.max(average_ds.cond_repr.isel(x=refx, y=refy+1).values)
    cond_repr_down = np.max(average_ds.cond_repr.isel(x=refx, y=refy-1).values)
    cond_repr = min(cond_repr_right, cond_repr_down, cond_repr_up, cond_repr_left)

    return v_c, w_c, vx_cc_tde, vy_cc_tde, pd.confidence, vx_2dca_tde, vy_2dca_tde, cond_repr

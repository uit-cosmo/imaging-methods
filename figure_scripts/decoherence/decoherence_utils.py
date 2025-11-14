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
repo_root = Path(__file__).resolve().parent.parent  # src → my_project
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
        "window": 60,
        "check_max": 1,
        "single_counting": True,
    },
    "gauss_fit": {"size_penalty": 5, "aspect_penalty": 0.2, "tilt_penalty": 0.2},
    "contouring": {"threshold_factor": 0.3, "com_smoothing": 10},
    "taud_estimation": {"cutoff": 1e6, "nperseg": 1e3},
}

T = 10000
Lx = 8
Ly = 8
nx = 8
ny = 8
dt = 0.1
bs = BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian)
K = 10000

vx_input = 1
vy_intput = 0
lx_input = 1
ly_input = 1


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

    distances_vector = contour_ds.center_of_mass.values - [
        average_ds.R.isel(x=refx, y=refy).item(),
        average_ds.Z.isel(x=refx, y=refy).item(),
    ]
    distances = np.sqrt((distances_vector**2).sum(axis=1))

    v_c, w_c = (
        velocity_ds.sel(time=contour_ds.time[distances < 1])
        .mean(dim="time", skipna=True)
        .values
    )

    eo = ve.EstimationOptions()
    eo.cc_options.cc_window = method_parameters["2dca"]["window"] * dt
    eo.cc_options.minimum_cc_value = 0
    pd = ve.estimate_velocities_for_pixel(refx, refy, ve.CModImagingDataInterface(ds))
    vx_cc_tde, vy_cc_tde = pd.vx, pd.vy

    vx_2dca_tde, vy_2dca_tde = im.get_3tde_velocities(average_ds.cond_av, refx, refy)
    cond_repr_right = np.max(average_ds.cond_repr.isel(x=refx + 1, y=refy).values)
    cond_repr_left = np.max(average_ds.cond_repr.isel(x=refx - 1, y=refy).values)
    cond_repr_up = np.max(average_ds.cond_repr.isel(x=refx, y=refy + 1).values)
    cond_repr_down = np.max(average_ds.cond_repr.isel(x=refx, y=refy - 1).values)
    cond_repr_neigh = max(cond_repr_right, cond_repr_down, cond_repr_up, cond_repr_left)
    cond_repr_ref = average_ds.cond_repr.isel(x=refx, y=refy).max().item()

    return (
        v_c,
        w_c,
        vx_cc_tde,
        vy_cc_tde,
        pd.confidence,
        vx_2dca_tde,
        vy_2dca_tde,
        cond_repr_neigh / cond_repr_ref,
    )


def make_decoherence_realization(rand_coeff):
    def blob_getter():
        return get_blob(
            amplitude=np.random.exponential(),
            vx=np.random.uniform(1 - rand_coeff, 1 + rand_coeff),
            vy=np.random.uniform(-rand_coeff, rand_coeff),
            posx=np.random.uniform(0, Lx),
            posy=np.random.uniform(0, Ly),
            lx=1,
            ly=1,
            t_init=np.random.uniform(0, T),
            bs=bs,
            theta=0,  # 0 but go to the != 0 branch in the blob.py function
        )

    return make_2d_realization(
        Lx,
        Ly,
        T,
        nx,
        ny,
        dt,
        K,
        vx=vx_input,
        vy=vy_intput,
        lx=1,
        ly=1,
        theta=0,
        bs=bs,
        blob_getter=blob_getter,
    )

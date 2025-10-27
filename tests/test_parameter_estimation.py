import imaging_methods as im
from test_utils import (
    make_2d_realization,
    get_blob,
    DeterministicBlobFactory,
    Model,
)
from blobmodel import BlobShapeEnum, BlobShapeImpl
import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt

Lx = 10
Ly = 10
nx = 1
ny = 1
dt = 0.01
theta = 0
bs = BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.exp)

refx, refy = 0, 0


def test_fit():
    average = xr.open_dataset("tests/test_average_gpi.nc")
    fig, ax = plt.subplots()
    im.plot_event_with_fit(average, None, 6, 6, ax)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$Z$")
    plt.savefig("example_fit.pdf", bbox_inches="tight")
    plt.show()

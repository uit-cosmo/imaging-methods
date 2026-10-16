"""The decorrelation tests' synthetic record, its settings and its bank, made once a
session (see pipeline_fixture.py)."""

import pytest

import pipeline_fixture as fixture
from imaging_methods import decorrelation


@pytest.fixture(scope="session")
def record():
    return fixture.load_record()


@pytest.fixture(scope="session")
def settings():
    """The fixture's settings, ``(averages, tracking, tde, blobs)``."""
    return fixture.settings()


@pytest.fixture(scope="session")
def dead(record):
    return decorrelation.dead_mask(record)


@pytest.fixture(scope="session")
def fresh(record, settings, dead):
    """average() at every live reference."""
    return {
        (x, y): decorrelation.average(record, settings[0], x, y)
        for x, y in decorrelation.references(dead.values)
    }


@pytest.fixture(scope="session")
def bank(record, settings, dead, fresh):
    return decorrelation.stack(fresh, record, settings[0], dead)


@pytest.fixture(scope="session")
def fields_out(record, settings, dead, bank):
    averages, tracking, tde, _ = settings
    return decorrelation.fields(
        record,
        lambda x, y: decorrelation.at(bank, x, y),
        averages,
        tracking,
        tde,
        dead,
    )


@pytest.fixture(scope="session")
def blobs_out(record, settings, dead, bank):
    averages, tracking, _, blob_settings = settings
    return decorrelation.blobs(
        record,
        lambda x, y: decorrelation.at(bank, x, y),
        averages,
        tracking.neighbour_step,
        blob_settings,
        dead,
    )

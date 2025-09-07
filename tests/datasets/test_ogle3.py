#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.datasets.ogle3 import (
    OGLE_CATALOG_BASE_URL,
    OGLE_CATALOG_PATH,
    _get_OGLE3_data_home,
    fetch_OGLE3,
    load_OGLE3_catalog,
)

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_load_OGLE3_catalog(mocker):
    df = pd.DataFrame({"# ID": [1, 2, 3]})

    mocker.patch("pandas.read_table", return_value=df)

    BZ2File = mocker.patch("bz2.BZ2File")
    BZ2File.return_value.__enter__ = mocker.Mock()

    assert "ID" not in df.columns
    assert "# ID" in df.columns

    load_OGLE3_catalog()

    assert "ID" in df.columns
    assert "# ID" not in df.columns

    BZ2File.assert_called_with(OGLE_CATALOG_PATH)


@pytest.mark.slow
@pytest.mark.parametrize(
    ["read_data", "expected_time", "expected_magnitude", "expected_error"],
    [
        ([1, 10, 0.1], [1], [10], [0.1]),
        (
            [[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]],
            [1, 2, 3],
            [10, 20, 30],
            [0.1, 0.2, 0.3],
        ),
    ],
)
def test_fetch_OGLE3(
    mocker, read_data, expected_time, expected_magnitude, expected_error
):
    store_path = _get_OGLE3_data_home(None)

    fetch = mocker.patch("feets.datasets.base.fetch")
    mocker.patch("numpy.loadtxt", return_value=np.array(read_data))

    ds = fetch_OGLE3("OGLE-BLG-LPV-232377")

    np.testing.assert_equal(ds._id, "OGLE-BLG-LPV-232377")
    np.testing.assert_equal(set(ds.bands), {"I", "V"})

    np.testing.assert_equal(ds.data.I.time, np.array(expected_time))
    np.testing.assert_equal(ds.data.I.magnitude, np.array(expected_magnitude))
    np.testing.assert_equal(ds.data.I.error, np.array(expected_error))

    np.testing.assert_equal(ds.data.V.time, np.array(expected_time))
    np.testing.assert_equal(ds.data.V.magnitude, np.array(expected_magnitude))
    np.testing.assert_equal(ds.data.V.error, np.array(expected_error))

    called_with = {call.args for call in fetch.mock_calls}
    np.testing.assert_equal(
        set(called_with),
        {
            (
                f"{OGLE_CATALOG_BASE_URL}/I/77/OGLE-BLG-LPV-232377.dat",
                store_path / "OGLE-BLG-LPV-232377.I.dat",
            ),
            (
                f"{OGLE_CATALOG_BASE_URL}/V/77/OGLE-BLG-LPV-232377.dat",
                store_path / "OGLE-BLG-LPV-232377.V.dat",
            ),
        },
    )


@pytest.mark.slow
def test_fetch_OGLE3_file_not_found(mocker):
    cat = load_OGLE3_catalog()
    oid = np.random.choice(cat.ID)

    mocker.patch("feets.datasets.base.fetch")
    mocker.patch(
        "numpy.loadtxt",
        side_effect=FileNotFoundError,
    )

    with pytest.raises(FileNotFoundError):
        fetch_OGLE3(oid, download_if_missing=False)


@pytest.mark.slow
def test_fetch_OGLE3_id_not_found():
    oid = "invalid_id"

    with pytest.raises(ValueError):
        fetch_OGLE3(oid)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe; Clariá, Felipe
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
def test_fetch_OGLE3(mocker):
    store_path = _get_OGLE3_data_home(None)
    cat = load_OGLE3_catalog()
    oid = np.random.choice(cat.ID)

    url_I = f"{OGLE_CATALOG_BASE_URL}/I/{oid[-2:]}/{oid}.dat"
    url_V = f"{OGLE_CATALOG_BASE_URL}/V/{oid[-2:]}/{oid}.dat"

    file_path_I = store_path / f"{oid}.I.dat"
    file_path_V = store_path / f"{oid}.V.dat"

    fetch = mocker.patch("feets.datasets.base.fetch")
    mocker.patch(
        "numpy.loadtxt",
        return_value=np.array([[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]]),
    )

    ds = fetch_OGLE3(oid)

    np.testing.assert_equal(ds._id, oid)
    np.testing.assert_equal(ds.bands, ["I", "V"])

    np.testing.assert_equal(ds.data.I.time, np.array([1, 2, 3]))
    np.testing.assert_equal(ds.data.I.magnitude, np.array([10, 20, 30]))
    np.testing.assert_equal(ds.data.I.error, np.array([0.1, 0.2, 0.3]))

    np.testing.assert_equal(ds.data.V.time, np.array([1, 2, 3]))
    np.testing.assert_equal(ds.data.V.magnitude, np.array([10, 20, 30]))
    np.testing.assert_equal(ds.data.V.error, np.array([0.1, 0.2, 0.3]))

    called_with = {call.args for call in fetch.mock_calls}
    np.testing.assert_equal(
        called_with, {(url_I, file_path_I), (url_V, file_path_V)}
    )


@pytest.mark.slow
def test_fetch_OGLE3_file_not_found(mocker):
    cat = load_OGLE3_catalog()
    oid = np.random.choice(cat.ID)

    mocker.patch(
        "numpy.loadtxt",
        side_effect=[
            np.array([[1, 2, 3], [10, 20, 30], [0.1, 0.1, 0.1]]),
            FileNotFoundError,
        ],
    )

    with pytest.raises(FileNotFoundError):
        fetch_OGLE3(oid, download_if_missing=False)


@pytest.mark.slow
def test_fetch_OGLE3_id_not_found():
    oid = "invalid_id"

    with pytest.raises(ValueError):
        fetch_OGLE3(oid)

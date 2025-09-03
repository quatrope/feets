#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.datasets.macho import (
    DATA_PATH,
    MACHO_EXAMPLE_ID,
    available_MACHO_lc,
    load_MACHO,
    load_MACHO_example,
)

import numpy as np


# =============================================================================
# TESTS
# =============================================================================


def test_available_MACHO_lc(mocker):
    mock_listdir = mocker.patch(
        "feets.datasets.macho.os.listdir",
        return_value=[
            "lc_1_3444_614.tar.bz2",
            "lc_1_3444_615.tar.bz2",
            "lc_2_1111_222.tar.bz2",
        ],
    )

    result = available_MACHO_lc()
    expected = ["lc_1_3444_614", "lc_1_3444_615", "lc_2_1111_222"]

    np.testing.assert_equal(result, expected)
    mock_listdir.assert_called_once_with(DATA_PATH)


def test_load_MACHO_example(mocker):
    path = DATA_PATH / f"{MACHO_EXAMPLE_ID}.tar.bz2"

    file_path_R = f"{MACHO_EXAMPLE_ID}.R.mjd"
    file_path_B = f"{MACHO_EXAMPLE_ID}.B.mjd"

    tarfile_open = mocker.patch("feets.datasets.macho.tarfile.open")
    extractfile = mocker.Mock()
    tarfile_open.return_value.__enter__ = mocker.Mock(
        return_value=mocker.Mock(extractfile=extractfile)
    )

    mocker.patch(
        "numpy.loadtxt",
        return_value=np.array([[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]]),
    )

    ds = load_MACHO_example()

    np.testing.assert_equal(ds._id, MACHO_EXAMPLE_ID)
    np.testing.assert_equal(ds.bands, ["R", "B"])

    np.testing.assert_equal(ds.data.R.time, np.array([1, 2, 3]))
    np.testing.assert_equal(ds.data.R.magnitude, np.array([10, 20, 30]))
    np.testing.assert_equal(ds.data.R.error, np.array([0.1, 0.2, 0.3]))

    np.testing.assert_equal(ds.data.B.time, np.array([1, 2, 3]))
    np.testing.assert_equal(ds.data.B.magnitude, np.array([10, 20, 30]))
    np.testing.assert_equal(ds.data.B.error, np.array([0.1, 0.2, 0.3]))

    tarfile_open.assert_called_once_with(path, mode="r:bz2")

    called_with = {call.args for call in extractfile.mock_calls}
    np.testing.assert_equal(called_with, {(file_path_R,), (file_path_B,)})


def test_load_MACHO(mocker):
    cat = available_MACHO_lc()
    mid = np.random.choice(cat)
    path = DATA_PATH / f"{mid}.tar.bz2"

    file_path_R = f"{mid}.R.mjd"
    file_path_B = f"{mid}.B.mjd"

    tarfile_open = mocker.patch("feets.datasets.macho.tarfile.open")
    extractfile = mocker.Mock()
    tarfile_open.return_value.__enter__ = mocker.Mock(
        return_value=mocker.Mock(extractfile=extractfile)
    )

    mocker.patch(
        "numpy.loadtxt",
        return_value=np.array([[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]]),
    )

    ds = load_MACHO(mid)

    np.testing.assert_equal(ds._id, mid)
    np.testing.assert_equal(ds.bands, ["R", "B"])

    np.testing.assert_equal(ds.data.R.time, np.array([1, 2, 3]))
    np.testing.assert_equal(ds.data.R.magnitude, np.array([10, 20, 30]))
    np.testing.assert_equal(ds.data.R.error, np.array([0.1, 0.2, 0.3]))

    np.testing.assert_equal(ds.data.B.time, np.array([1, 2, 3]))
    np.testing.assert_equal(ds.data.B.magnitude, np.array([10, 20, 30]))
    np.testing.assert_equal(ds.data.B.error, np.array([0.1, 0.2, 0.3]))

    tarfile_open.assert_called_once_with(path, mode="r:bz2")

    called_with = {call.args for call in extractfile.mock_calls}
    np.testing.assert_equal(called_with, {(file_path_R,), (file_path_B,)})

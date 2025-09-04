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

import feets
from feets.preprocess import align, remove_noise

import numpy as np

import pytest

import tests.conftest as conftest


# =============================================================================
# CLASSES
# =============================================================================


@pytest.fixture
def MACHO_LC():
    lc = feets.datasets.load_MACHO_example()
    lc = {
        "time": lc.data.R.time,
        "magnitude": lc.data.R.magnitude,
        "error": lc.data.R.error,
        "time2": lc.data.B.time,
        "magnitude2": lc.data.B.magnitude,
        "error2": lc.data.B.error,
    }
    return lc


@pytest.fixture
def FATS_MACHO_LC_remove_noise_result():
    path = conftest.TEST_DATASET_PATH / "FATS_preprc.npz"
    lc = {}
    with np.load(path) as npz:
        lc = {
            "time": npz["time"],
            "time2": npz["time2"],
            "magnitude": npz["mag"],
            "magnitude2": npz["mag2"],
            "error": npz["error"],
            "error2": npz["error2"],
        }
    return lc


@pytest.fixture
def FATS_MACHO_LC_remove_noise_aligned():
    path = conftest.TEST_DATASET_PATH / "FATS_aligned.npz"
    lc = {}
    with np.load(path) as npz:
        lc = {
            "time": npz["aligned_time"],
            "magnitude": npz["aligned_mag"],
            "magnitude2": npz["aligned_mag2"],
            "error": npz["aligned_error"],
            "error2": npz["aligned_error2"],
        }
    return lc


def test_FATS2feets_remove_noise(MACHO_LC, FATS_MACHO_LC_remove_noise_result):
    time_clean, magnitude_clean, error_clean = remove_noise(
        MACHO_LC["time"], MACHO_LC["magnitude"], MACHO_LC["error"]
    )
    time2_clean, magnitude2_clean, error2_clean = remove_noise(
        MACHO_LC["time2"], MACHO_LC["magnitude2"], MACHO_LC["error2"]
    )
    np.testing.assert_array_equal(
        time_clean, FATS_MACHO_LC_remove_noise_result["time"]
    )
    np.testing.assert_array_equal(
        time2_clean, FATS_MACHO_LC_remove_noise_result["time2"]
    )
    np.testing.assert_array_equal(
        magnitude_clean, FATS_MACHO_LC_remove_noise_result["magnitude"]
    )
    np.testing.assert_array_equal(
        magnitude2_clean, FATS_MACHO_LC_remove_noise_result["magnitude2"]
    )
    np.testing.assert_array_equal(
        error_clean, FATS_MACHO_LC_remove_noise_result["error"]
    )
    np.testing.assert_array_equal(
        error2_clean, FATS_MACHO_LC_remove_noise_result["error2"]
    )


def test_FATS2feets_align(MACHO_LC, FATS_MACHO_LC_remove_noise_aligned):
    time_clean, magnitude_clean, error_clean = remove_noise(
        MACHO_LC["time"], MACHO_LC["magnitude"], MACHO_LC["error"]
    )
    time2_clean, magnitude2_clean, error2_clean = remove_noise(
        MACHO_LC["time2"], MACHO_LC["magnitude2"], MACHO_LC["error2"]
    )

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(
        time=time_clean,
        time2=time2_clean,
        magnitude=magnitude_clean,
        magnitude2=magnitude2_clean,
        error=error_clean,
        error2=error2_clean,
    )

    np.testing.assert_array_equal(
        aligned_time, FATS_MACHO_LC_remove_noise_aligned["time"]
    )
    np.testing.assert_array_equal(
        aligned_magnitude, FATS_MACHO_LC_remove_noise_aligned["magnitude"]
    )
    np.testing.assert_array_equal(
        aligned_magnitude2, FATS_MACHO_LC_remove_noise_aligned["magnitude2"]
    )
    np.testing.assert_array_equal(
        aligned_error, FATS_MACHO_LC_remove_noise_aligned["error"]
    )
    np.testing.assert_array_equal(
        aligned_error2, FATS_MACHO_LC_remove_noise_aligned["error2"]
    )

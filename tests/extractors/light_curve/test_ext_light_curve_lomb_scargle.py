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

from feets.extractors.light_curve.ext_light_curve_lomb_scargle import (
    LightCurveLombScargle,
)

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 101
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_LightCurveLombScargle_extract(normal):
    # init extractor
    extractor = LightCurveLombScargle(
        peaks=2,
        resolution=20.0,
        max_freq_factor=2.0,
        nyquist="average",
        fast=True,
    )

    # simulate results
    time = np.linspace(0, 10, LC_LENGTH)
    magnitude = np.sin(2 * np.pi * time / 0.7) + 0.5 * np.cos(
        2 * np.pi * time / 3.3
    )
    results = extractor.extract(time=time, magnitude=magnitude)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = [
        [0.69896194, 3.31147541],  # Periodogram_Peaks
        [11.5355674, 3.20085143],  # Periodogram_S_to_N
    ]

    np.testing.assert_allclose(values, expected)

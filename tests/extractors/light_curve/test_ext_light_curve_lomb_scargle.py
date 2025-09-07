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

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 101
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_LightCurveLombScargle_extract():
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
    result = extractor.extract(time=time, magnitude=magnitude)

    # values by feature
    df = pd.DataFrame(result)

    # expected columns and mean values
    expected = pd.DataFrame(
        {
            "Periodogram_Peaks": [0.69896194, 3.31147541],
            "Periodogram_S_to_N": [11.5355674, 3.20085143],
        }
    )

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.columns))

    # check values
    np.testing.assert_allclose(df[expected.columns], expected)

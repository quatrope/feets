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

from feets.extractors.ext_median_amplitude import MedianAmplitude

import numpy as np

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_MedianAmplitude_extract():
    # init extractor
    extractor = MedianAmplitude()

    # simulate results
    lc = {"magnitude": np.arange(LC_LENGTH + 1)}
    results = extractor.extract(**lc)

    # values by feature
    series = pd.Series(results)

    # expected columns and mean values
    expected = pd.Series({"MedianAmplitude": 475.0})

    # check columns
    np.testing.assert_equal(set(series.index), set(expected.index))

    # check values
    np.testing.assert_allclose(series[expected.index], expected)

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

from feets.extractors.light_curve.ext_duration import Duration

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


def test_Duration_extract():
    # init extractor
    extractor = Duration()

    # simulate results
    lc = {"time": np.arange(LC_LENGTH + 1)}
    kwargs = extractor.prepare_extract(lc, {})
    results = extractor.extract(**kwargs)

    # values by feature
    series = pd.Series(results)

    # expected columns and mean values
    expected = pd.Series({"Duration": 1000})

    # check columns
    np.testing.assert_equal(set(series.index), set(expected.index))

    # check values
    np.testing.assert_allclose(series[expected.index], expected)

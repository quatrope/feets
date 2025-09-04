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

from feets.extractors.light_curve.ext_min_time_interval import MinTimeInterval

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_MinTimeInterval_extract():
    # init extractor
    extractor = MinTimeInterval()

    # simulate results
    lc = {"time": np.arange(LC_LENGTH)}
    kwargs = extractor.prepare_extract(lc, {})
    results = extractor.extract(**kwargs)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = 1  # MinTimeInterval
    np.testing.assert_allclose(values, expected)

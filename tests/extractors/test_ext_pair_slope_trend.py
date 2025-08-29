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

from feets.extractors.ext_pair_slope_trend import PairSlopeTrend

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
MAX_ITERS = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_PairSlopeTrend_extract(normal):
    extractor = PairSlopeTrend()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {"magnitude": normal(random=random, size=LC_LENGTH)}
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert mean is close to expected value
    expected = -0.0021333333333333343  # PairSlopeTrend
    np.testing.assert_allclose(values.mean(axis=0), expected)

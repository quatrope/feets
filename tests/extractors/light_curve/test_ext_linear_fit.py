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

from feets.extractors.light_curve.ext_linear_fit import LinearFit

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
def test_LinearFit_extract(normal):
    # init extractor
    extractor = LinearFit()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH),
            "magnitude": normal(random=random, size=LC_LENGTH),
            "error": normal(random=random, size=LC_LENGTH, loc=1, scale=0.008),
        }
        for _ in range(MAX_ITERS)
    ]
    kwargss = [extractor.prepare_extract(lc, {}) for lc in lcs]
    results = [extractor.extract(**kwargs) for kwargs in kwargss]

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert results are close to expected value
    expected = [
        0.000007242744636577737,  # LinearFit_Slope
        0.00010952101130705979,  # LinearFit_Sigma
        0.998946109918393,  # LinearFit_ReducedChi2
    ]

    np.testing.assert_allclose(values.mean(axis=0), expected)

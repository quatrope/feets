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

from feets.extractors.light_curve.ext_otsu_split import OtsuSplit

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
def test_OtsuSplit_extract(normal):
    # init extractor
    extractor = OtsuSplit()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {"magnitude": normal(random=random, size=LC_LENGTH)}
        for _ in range(MAX_ITERS)
    ]
    kwargss = [extractor.prepare_extract(lc, {}) for lc in lcs]
    results = [extractor.extract(**kwargs) for kwargs in kwargss]

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert results are close to expected value
    expected = [
        1.5981193899074284,  # OtsuMeanDiff
        0.6030745633913142,  # OtsuStdLower
        0.6024007987301051,  # OtsuStdUpper
        0.5003209999999997,  # OtsuLowerToAllRatio
    ]

    np.testing.assert_allclose(values.mean(axis=0), expected)

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

from feets.extractors.ext_car import CAR

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 100
MAX_ITERS = 100
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_CAR_extract(periodic, normal):
    # init extractor
    extractor = CAR()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH),
            "magnitude": periodic(random=random, size=LC_LENGTH),
            "error": normal(random=random, size=LC_LENGTH, loc=1, scale=0.008),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert mean is close to expected value
    expected = [
        0.008015313327483975,  # CAR_sigma
        0.6475047826376705,  # CAR_tau
        -0.11911673512729966,  # CAR_mean
    ]
    np.testing.assert_allclose(values.mean(axis=0), expected)

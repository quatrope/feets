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

import pandas as pd

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

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series(
        {
            "CAR_sigma": 0.008015313327483975,
            "CAR_tau": 0.6475047826376705,
            "CAR_mean": -0.11911673512729966,
        }
    )

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)

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

from feets.extractors.ext_weighted_beyond_n_std import WeightedBeyondNStd

import numpy as np

import pandas as pd

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


@pytest.mark.parametrize("nstd", [0, -1])
def test_WeightedBeyondNStd_non_positive_integer(nstd):
    with pytest.raises(ValueError):
        WeightedBeyondNStd(nstd=nstd)


def test_WeightedBeyondNStd_extract(normal):
    # init extractor
    extractor = WeightedBeyondNStd(nstd=1)

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "magnitude": normal(random=random, size=LC_LENGTH),
            "error": normal(random=random, size=LC_LENGTH, loc=1, scale=0.008),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"WeightedBeyondNStd": 0.317288})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)


def test_WeightedBeyondNStd_flatten_feature():
    # init extractor
    extractor = WeightedBeyondNStd(nstd=1)

    features = {"WeightedBeyondNStd": 0.317288, "test_feature": [1, 2, 3]}

    # flatten results
    flattened_results = {
        feature: extractor.flatten_feature(feature, value)
        for feature, value in features.items()
    }

    # check flattened results
    expected = {
        "WeightedBeyondNStd": {"WeightedBeyond1Std": 0.317288},
        "test_feature": {
            "test_feature_0": 1,
            "test_feature_1": 2,
            "test_feature_2": 3,
        },
    }
    np.testing.assert_equal(flattened_results, expected)

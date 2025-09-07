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

from feets.extractors.ext_percent_diff_percentile import PercentDiffPercentile

import numpy as np

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
MAX_ITERS = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_PercentDiffPercentile_extract(uniform):
    # init extractor
    extractor = PercentDiffPercentile(quantile=0.05)

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {"magnitude": uniform(random=random, size=LC_LENGTH)}
        for _ in range(MAX_ITERS)
    ]
    kwargss = [extractor.prepare_extract(lc, {}) for lc in lcs]
    results = [extractor.extract(**kwargs) for kwargs in kwargss]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"PercentDiffPercentile": 1.8006518831938725})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected)


def test_PercentDiffPercentile_flatten_feature():
    # init extractor
    extractor = PercentDiffPercentile(quantile=0.05)

    features = {
        "PercentDiffPercentile": 1.8006518831938725,
        "test_feature": [1, 2, 3],
    }

    # flatten results
    flattened_results = {
        feature: extractor.flatten_feature(feature, value)
        for feature, value in features.items()
    }

    # check flattened results
    expected = {
        "PercentDiffPercentile": {
            "PercentDiffPercentile_5": 1.8006518831938725
        },
        "test_feature": {
            "test_feature_0": 1,
            "test_feature_1": 2,
            "test_feature_2": 3,
        },
    }
    np.testing.assert_equal(flattened_results, expected)

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

from feets.extractors.ext_autocor_length import AutocorLength

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


def test_AutocorLength_extract(normal):
    # init extractor
    extractor = AutocorLength(nlags=10)

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {"magnitude": normal(random=random, size=LC_LENGTH)}
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"Autocor_length": 1})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected)


def test_AutocorLength_extract_long_correlation(periodic):
    # init extractor
    extractor = AutocorLength(nlags=100)

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lc = {
        "magnitude": periodic(
            random=random, size=LC_LENGTH, period=LC_LENGTH * 10
        )
    }
    result = extractor.extract(**lc)

    np.testing.assert_equal(result, {"Autocor_length": 220})

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

from feets.extractors.ext_stetson import StetsonJ, StetsonKAC, StetsonL

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
LC_LENGTH_SHORT = 100
MAX_ITERS = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_StetsonJ_extract(normal):
    # init extractor
    extractor = StetsonJ()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "aligned_magnitude": normal(random=random, size=LC_LENGTH),
            "aligned_magnitude2": normal(random=random, size=LC_LENGTH),
            "aligned_error": normal(
                random=random, size=LC_LENGTH, loc=1, scale=0.008
            ),
            "aligned_error2": normal(
                random=random, size=LC_LENGTH, loc=1, scale=0.008
            ),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"StetsonJ": 0.000389878261606318})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)


@pytest.mark.slow
def test_StetsonKAC_extract(normal):
    # init extractor
    extractor = StetsonKAC()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH_SHORT),
            "magnitude": normal(random=random, size=LC_LENGTH_SHORT),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"StetsonK_AC": 0.6583779})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)


def test_StetsonL_extract(normal):
    # init extractor
    extractor = StetsonL()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "aligned_magnitude": normal(random=random, size=LC_LENGTH),
            "aligned_magnitude2": normal(random=random, size=LC_LENGTH),
            "aligned_error": normal(
                random=random, size=LC_LENGTH, loc=1, scale=0.008
            ),
            "aligned_error2": normal(
                random=random, size=LC_LENGTH, loc=1, scale=0.008
            ),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"StetsonL": 0.00030183305778540346})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)

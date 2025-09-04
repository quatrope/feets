#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.ext_stetson import StetsonJ, StetsonKAC, StetsonL

import numpy as np

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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_StetsonJ_extract(normal):
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

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert mean is close to expected value
    expected = 0.000389878261606318  # StetsonJ

    np.testing.assert_allclose(values.mean(axis=0), expected)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_StetsonKAC_extract(normal):
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

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert mean is close to expected value
    expected = 0.6583779  # StetsonK_AC

    np.testing.assert_allclose(values.mean(axis=0), expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_StetsonL_extract(normal):
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

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert mean is close to expected value
    expected = 0.00030183305778540346  # StetsonL

    np.testing.assert_allclose(values.mean(axis=0), expected)

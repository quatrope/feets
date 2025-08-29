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

from feets.extractors.ext_eta_color import EtaColor

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
def test_EtaColor_extract(normal):
    # init extractor
    extractor = EtaColor()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "aligned_time": np.arange(LC_LENGTH),
            "aligned_magnitude": normal(random=random, size=LC_LENGTH),
            "aligned_magnitude2": normal(random=random, size=LC_LENGTH),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # transform results into ndarray
    values = np.array([list(result.values()) for result in results])

    # assert mean is close to expected value
    expected = 1.9941614810240644  # Eta_color
    np.testing.assert_allclose(values.mean(axis=0), expected)

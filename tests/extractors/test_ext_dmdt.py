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

from feets.extractors.ext_dmdt import DeltamDeltat

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
MAX_ITERS = 50
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_DeltamDeltat_extract(normal):
    # init extractor
    extractor = DeltamDeltat()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH),
            "magnitude": normal(random=random, size=LC_LENGTH),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # transform results into ndarray
    values = np.array(
        [np.sum(list(result["DeltamDeltat"].values())) for result in results]
    )

    # assert mean is close to expected value
    expected = 425.86
    np.testing.assert_allclose(values.mean(axis=0), expected)

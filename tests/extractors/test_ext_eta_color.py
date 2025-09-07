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

from feets.extractors.ext_eta_color import EtaColor

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

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series({"Eta_color": 1.9941614810240644})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)

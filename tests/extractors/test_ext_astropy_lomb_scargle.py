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

from feets.extractors.ext_astropy_lomb_scargle import AstropyLombScargle

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_ITERS = 100
LC_LENGTH = 100
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.slow
def test_AstropyLombScargle_extract(periodic):
    # init extractor
    lscargle_kwds = {
        "autopower_kwds": {
            "normalization": "standard",
            "nyquist_factor": 1,
        }
    }
    extractor = AstropyLombScargle(lscargle_kwds=lscargle_kwds, nperiods=2)

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH),
            "magnitude": periodic(random=random, size=LC_LENGTH, period=20),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    dfs = [pd.DataFrame(result) for result in results]
    df = pd.concat(dfs, keys=range(len(dfs)))

    # expected columns and mean values
    expected = pd.DataFrame(
        {
            "PeriodLS": [18.808016760831848, 19.27779111644658],
            "Period_fit": [0, 0],
            "Psi_CS": [0.25401119954358997, 0.2672274175209012],
            "Psi_eta": [0.4105175583550706, 0.06377275549870381],
        }
    )

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.columns))

    # check means
    means = df.groupby(level=1).mean()[expected.columns]
    np.testing.assert_allclose(means, expected, rtol=2e-2, atol=1e-3)

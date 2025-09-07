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
    extractor = AstropyLombScargle(lscargle_kwds=lscargle_kwds, nperiods=3)

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
            "PeriodLS": [
                20.26250834469941,
                18.771094343627034,
                19.27779111644655,
            ],
            "Period_fit": [
                1.4306433603192435e-11,
                7.701492122755767e-13,
                1.1292193858099717e-13,
            ],
            "Psi_CS": [
                0.23181927251239123,
                0.2555247224341927,
                0.269163774493831,
            ],
            "Psi_eta": [
                0.9003366875414929,
                0.40641056227603306,
                0.06352767993482751,
            ],
        }
    )

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.columns))

    # check means
    means = df.groupby(level=1).mean()[expected.columns]
    np.testing.assert_allclose(means.to_numpy(), expected.to_numpy())

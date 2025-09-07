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

from feets.extractors.ext_bazin_fit import BazinFit

import numpy as np

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 101
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_BazinFit_extract():
    # init extractor
    extractor = BazinFit(algorithm="mcmc")

    # simulate results
    time = np.linspace(0, 10, LC_LENGTH)
    flux = 1 + (time - 3) ** 2
    flux_error = np.sqrt(flux)
    results = extractor.extract(time=time, flux=flux, flux_error=flux_error)

    # values by feature
    series = pd.Series(results)

    # expected columns and mean values
    expected = pd.Series(
        {
            "BazinFit_Amplitude": 53.83059247046932,
            "BazinFit_Baseline": 1.7157666084967467,
            "BazinFit_ReferenceTime": 7.935474996154883,
            "BazinFit_RiseTime": 0.917867417470989,
            "BazinFit_FallTime": 13.234191886674147,
            "BazinFit_ReducedChi2": 0.4123428662618001,
        }
    )

    # check columns
    np.testing.assert_equal(set(series.index), set(expected.index))

    # check values
    np.testing.assert_allclose(series[expected.index], expected)

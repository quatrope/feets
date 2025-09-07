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

from feets.extractors.light_curve.ext_linexp_fit import LinexpFit

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


def test_LinexpFit_extract():
    # init extractor
    extractor = LinexpFit(algorithm="mcmc")

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
            "LinexpFit_Amplitude": 62.559756035926725,
            "LinexpFit_Baseline": -0.983590724517521,
            "LinexpFit_ReferenceTime": 1.166120371901874,
            "LinexpFit_FallTime": 24.53696765735986,
            "LinexpFit_ReducedChi2": 3.5270992947607924,
        }
    )

    # check columns
    np.testing.assert_equal(set(series.index), set(expected.index))

    # check values
    np.testing.assert_allclose(series[expected.index], expected, rtol=1e-2)

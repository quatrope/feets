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

from feets.extractors.light_curve.ext_linexp_fit import LinexpFit

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 101
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_LinexpFit_extract(normal):
    # init extractor
    extractor = LinexpFit(algorithm="mcmc")

    # simulate results
    time = np.linspace(0, 10, LC_LENGTH)
    flux = 1 + (time - 3) ** 2
    flux_error = np.sqrt(flux)
    results = extractor.extract(time=time, flux=flux, flux_error=flux_error)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = [
        62.559756035926725,  # LinexpFit_Amplitude
        -0.983590724517521,  # LinexpFit_Baseline
        1.166120371901874,  # LinexpFit_ReferenceTime
        24.53696765735986,  # LinexpFit_FallTime
        3.5270992947607924,  # LinexpFit_ReducedChi2
    ]

    np.testing.assert_allclose(values, expected)

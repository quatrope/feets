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

from feets.extractors.light_curve.ext_bazin_fit import BazinFit

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
def test_BazinFit_extract(normal):
    # init extractor
    extractor = BazinFit(algorithm="mcmc")

    # simulate results
    time = np.linspace(0, 10, LC_LENGTH)
    flux = 1 + (time - 3) ** 2
    flux_error = np.sqrt(flux)
    results = extractor.extract(time=time, flux=flux, flux_error=flux_error)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = [
        53.83059247046932,  # BazinFit_Amplitude
        1.7157666084967467,  # BazinFit_Baseline
        7.935474996154883,  # BazinFit_ReferenceTime
        0.917867417470989,  # BazinFit_RiseTime
        13.234191886674147,  # BazinFit_FallTime
        0.4123428662618001,  # BazinFit_ReducedChi2
    ]
    np.testing.assert_allclose(values, expected)

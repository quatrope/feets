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

from feets.extractors.light_curve.ext_villar_fit import VillarFit

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
def test_VillarFit_extract(normal):
    # init extractor
    extractor = VillarFit(algorithm="mcmc")

    # simulate results
    time = np.linspace(0, 10, LC_LENGTH)
    flux = 1 + (time - 3) ** 2
    flux_error = np.sqrt(flux)
    results = extractor.extract(time=time, flux=flux, flux_error=flux_error)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = [
        31.212447910981073,  # BazinFit_Amplitude
        1.9050126926568645,  # BazinFit_Baseline
        10.991797799738695,  # BazinFit_ReferenceTime
        1.0539717427724313,  # BazinFit_RiseTime
        7.7763331798464925,  # BazinFit_FallTime
        0.05014202207880292,  # VillarFit_PlateauRelAmplitude
        0.01307190643773739,  # VillarFit_PlateauDuration
        0.4981160671152967,  # BazinFit_ReducedChi2
    ]

    np.testing.assert_allclose(values, expected)

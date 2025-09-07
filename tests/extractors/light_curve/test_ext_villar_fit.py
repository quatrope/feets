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

from feets.extractors.light_curve.ext_villar_fit import VillarFit

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


def test_VillarFit_extract():
    # init extractor
    extractor = VillarFit(algorithm="mcmc")

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
            "VillarFit_Amplitude": 31.212447910981073,
            "VillarFit_Baseline": 1.9050126926568645,
            "VillarFit_ReferenceTime": 10.991797799738695,
            "VillarFit_RiseTime": 1.0539717427724313,
            "VillarFit_FallTime": 7.7763331798464925,
            "VillarFit_PlateauRelAmplitude": 0.05014202207880292,
            "VillarFit_PlateauDuration": 0.01307190643773739,
            "VillarFit_ReducedChi2": 0.4981160671152967,
        }
    )

    # check columns
    np.testing.assert_equal(set(series.index), set(expected.index))

    # check values
    np.testing.assert_allclose(series[expected.index], expected)

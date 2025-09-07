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

from feets.extractors.ext_fourier_components import FourierComponents

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
MAX_ITERS = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.slow
def test_FourierComponents_extract(normal):
    # init extractor
    lscargle_kwds = {
        "autopower_kwds": {
            "normalization": "standard",
            "nyquist_factor": 1,
        }
    }
    extractor = FourierComponents(lscargle_kwds=lscargle_kwds)

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

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series(
        {
            "Freq1_harmonics_amplitude_0": 0.1652338993415243,
            "Freq1_harmonics_amplitude_1": 0.055916832048932705,
            "Freq1_harmonics_amplitude_2": 0.057685163606276776,
            "Freq1_harmonics_amplitude_3": 0.05612955901457781,
            "Freq2_harmonics_amplitude_0": 0.11889568939927345,
            "Freq2_harmonics_amplitude_1": 0.05363088219338509,
            "Freq2_harmonics_amplitude_2": 0.05491345244237138,
            "Freq2_harmonics_amplitude_3": 0.05497801601685742,
            "Freq3_harmonics_amplitude_0": 0.05025187006485428,
            "Freq3_harmonics_amplitude_1": 0.03426252688615669,
            "Freq3_harmonics_amplitude_2": 0.04760288262267428,
            "Freq3_harmonics_amplitude_3": 0.05333792205363982,
            "Freq1_harmonics_rel_phase_0": 0.0,
            "Freq1_harmonics_rel_phase_1": -0.00432965069954101,
            "Freq1_harmonics_rel_phase_2": 0.004152429363126274,
            "Freq1_harmonics_rel_phase_3": -0.020219906186345532,
            "Freq2_harmonics_rel_phase_0": 0.0,
            "Freq2_harmonics_rel_phase_1": 0.00446796934645764,
            "Freq2_harmonics_rel_phase_2": 0.05022551988370977,
            "Freq2_harmonics_rel_phase_3": 0.055703688201031144,
            "Freq3_harmonics_rel_phase_0": 0.0,
            "Freq3_harmonics_rel_phase_1": 0.026434380371180073,
            "Freq3_harmonics_rel_phase_2": 0.0110188439207121,
            "Freq3_harmonics_rel_phase_3": -0.013173892905062655,
        }
    )

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected)

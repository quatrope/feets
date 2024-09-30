#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_fourier_components

import numpy as np


import pytest


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning", "ignore::scipy.optimize.OptimizeWarning"
)
def test_FourierComponents_extract(normal_light_curve):
    # create the extractor
    lscargle_kwds = {
        "autopower_kwds": {
            "normalization": "standard",
            "nyquist_factor": 1,
        }
    }
    extractor = ext_fourier_components.FourierComponents(
        lscargle_kwds=lscargle_kwds
    )
    features = [
        "Freq1_harmonics_amplitude_0",
        "Freq1_harmonics_amplitude_1",
        "Freq1_harmonics_amplitude_2",
        "Freq1_harmonics_amplitude_3",
        "Freq2_harmonics_amplitude_0",
        "Freq2_harmonics_amplitude_1",
        "Freq2_harmonics_amplitude_2",
        "Freq2_harmonics_amplitude_3",
        "Freq3_harmonics_amplitude_0",
        "Freq3_harmonics_amplitude_1",
        "Freq3_harmonics_amplitude_2",
        "Freq3_harmonics_amplitude_3",
        "Freq1_harmonics_rel_phase_0",
        "Freq1_harmonics_rel_phase_1",
        "Freq1_harmonics_rel_phase_2",
        "Freq1_harmonics_rel_phase_3",
        "Freq2_harmonics_rel_phase_0",
        "Freq2_harmonics_rel_phase_1",
        "Freq2_harmonics_rel_phase_2",
        "Freq2_harmonics_rel_phase_3",
        "Freq3_harmonics_rel_phase_0",
        "Freq3_harmonics_rel_phase_1",
        "Freq3_harmonics_rel_phase_2",
        "Freq3_harmonics_rel_phase_3",
    ]

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    sims = 1000
    size = 1000

    time = np.arange(size)
    values = np.empty([sims, 24])
    for idx in range(sims):
        lc = normal_light_curve(random=random, size=size, data=["magnitude"])
        lc["time"] = time

        results = extractor.extract(**lc)
        for index, feature in enumerate(features):
            values[idx, index] = results[feature]

    # test
    expected = [
        0.1652338991544277,
        0.05591683187222307,
        0.05768516373653064,
        0.05612955913340939,
        0.11889568955382276,
        0.05363088201255229,
        0.054913452901722955,
        0.054978015801523596,
        0.05025187056657103,
        0.034262526706087856,
        0.04760288280004671,
        0.05333792289571088,
        0.0,
        -0.0043296726801579765,
        0.004152429923662321,
        -0.02021990399751593,
        0.0,
        0.004467969120217738,
        0.05022552461794953,
        0.05570368699243758,
        0.0,
        0.026434353542337342,
        0.011018813392927828,
        -0.013173908272044727,
    ]
    np.testing.assert_allclose(values.mean(axis=0), expected)

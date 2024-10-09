#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_flux_percentile_ratio

import numpy as np


def test_FluxPercentileRatioMid20_extract(normal_light_curve):
    # create the extractor
    extractor = ext_flux_percentile_ratio.FluxPercentileRatioMid20()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["FluxPercentileRatioMid20"]

    np.testing.assert_allclose(values.mean(), 0.15400018087922435)


def test_FluxPercentileRatioMid35_extract(normal_light_curve):
    # create the extractor
    extractor = ext_flux_percentile_ratio.FluxPercentileRatioMid35()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["FluxPercentileRatioMid35"]

    np.testing.assert_allclose(values.mean(), 0.2756448446910934)


def test_FluxPercentileRatioMid50_extract(normal_light_curve):
    # create the extractor
    extractor = ext_flux_percentile_ratio.FluxPercentileRatioMid50()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["FluxPercentileRatioMid50"]

    np.testing.assert_allclose(values.mean(), 0.4095911599833594)


def test_FluxPercentileRatioMid65_extract(normal_light_curve):
    # create the extractor
    extractor = ext_flux_percentile_ratio.FluxPercentileRatioMid65()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["FluxPercentileRatioMid65"]

    np.testing.assert_allclose(values.mean(), 0.5674591670489371)


def test_FluxPercentileRatioMid80_extract(normal_light_curve):
    # create the extractor
    extractor = ext_flux_percentile_ratio.FluxPercentileRatioMid80()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["FluxPercentileRatioMid80"]

    np.testing.assert_allclose(values.mean(), 0.7785129402100429)

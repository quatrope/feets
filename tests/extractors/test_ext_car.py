#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_car

import numpy as np

import pytest


@pytest.mark.slow
def test_CAR_extract(periodic_light_curve):
    # create the extractor
    extractor = ext_car.CAR()
    features = ["CAR_mean", "CAR_sigma", "CAR_tau"]

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    sims = 100
    size = 100

    time = np.arange(size)
    values = np.empty([sims, 3])
    for idx in range(sims):
        lc = periodic_light_curve(
            random=random,
            size=size,
            data=["magnitude"],
        )
        lc["time"] = time
        lc["error"] = random.normal(loc=1, scale=0.008, size=size)

        results = extractor.extract(**lc)
        for index, feature in enumerate(features):
            values[idx, index] = results[feature]

    # test
    expected = [
        -0.11888100485725793,
        0.008015313327483975,
        0.6470569371786853,
    ]
    np.testing.assert_allclose(values.mean(axis=0), expected)

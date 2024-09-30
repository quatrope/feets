#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_linear_trend

import numpy as np


def test_LinearTrend_extract(normal_light_curve):
    # create the extractor
    extractor = ext_linear_trend.LinearTrend()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=10000, data=["magnitude"])
        lc["time"] = np.arange(10000)
        values[idx] = extractor.extract(**lc)["LinearTrend"]

    np.testing.assert_allclose(values.mean(), 5.093531729348368e-08)

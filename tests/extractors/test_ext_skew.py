#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_skew

import numpy as np


def test_Skew_extract(normal_light_curve):
    # create the extractor
    extractor = ext_skew.Skew()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["Skew"]

    np.testing.assert_allclose(values.mean(), -0.0012018544388092045)

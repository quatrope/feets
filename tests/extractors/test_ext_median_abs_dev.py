#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_median_abs_dev

import numpy as np


def test_MedianAbsDev_extract(normal_light_curve):
    # create the extractor
    extractor = ext_median_abs_dev.MedianAbsDev()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["MedianAbsDev"]

    np.testing.assert_allclose(values.mean(), 0.6735277130207087)

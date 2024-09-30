#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_q31

import numpy as np


def test_Q31_extract(normal_light_curve):
    # create the extractor
    extractor = ext_q31.Q31()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["Q31"]

    np.testing.assert_allclose(values.mean(), 1.3462968653954615)


def test_Q31Color_extract(normal_light_curve):
    # create the extractor
    extractor = ext_q31.Q31Color()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=["aligned_magnitude", "aligned_magnitude2"],
        )
        values[idx] = extractor.extract(**lc)["Q31_color"]

    np.testing.assert_allclose(values.mean(), 1.9017418802665758)

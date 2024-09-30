#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_beyond1_std

import numpy as np


def test_Beyond1Std_extract(normal_light_curve):
    # create the extractor
    extractor = ext_beyond1_std.Beyond1Std()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=["magnitude", "error"],
            error_loc=1,
            error_scale=0.008,
        )
        values[idx] = extractor.extract(**lc)["Beyond1Std"]

    # test!
    np.testing.assert_allclose(values.mean(), 0.317288)

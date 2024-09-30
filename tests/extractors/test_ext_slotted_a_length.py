#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import ext_slotted_a_length

import numpy as np

import pytest


@pytest.mark.slow
def test_SlottedA_length_extract(normal_light_curve):
    # create the extractor
    extractor = ext_slotted_a_length.SlottedA_length()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        lc["time"] = np.arange(1000)
        values[idx] = extractor.extract(**lc)["SlottedA_length"]

    np.testing.assert_allclose(values.mean(), 1.0)

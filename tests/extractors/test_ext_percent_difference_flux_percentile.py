#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

from feets.extractors import (
    ext_percent_difference_flux_percentile as ext_percent_diff_flux_percentile,
)


import numpy as np


def test_PercentDifferenceFluxPercentile_extract(uniform_light_curve):
    # create the extractor
    extractor = (
        ext_percent_diff_flux_percentile.PercentDifferenceFluxPercentile()
    )

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = uniform_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)[
            "PercentDifferenceFluxPercentile"
        ]

    np.testing.assert_allclose(values.mean(), 1.80068961898508)

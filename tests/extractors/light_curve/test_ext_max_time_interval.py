#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.light_curve.ext_max_time_interval import MaxTimeInterval

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_MaxTimeInterval_extract():
    # init extractor
    extractor = MaxTimeInterval()

    # simulate results
    lc = {"time": np.arange(LC_LENGTH)}
    kwargs = extractor.prepare_extract(lc, {})
    results = extractor.extract(**kwargs)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = 1  # MaxTimeInterval
    np.testing.assert_allclose(values, expected)

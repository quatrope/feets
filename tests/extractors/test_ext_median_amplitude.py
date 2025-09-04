#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.ext_median_amplitude import MedianAmplitude

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
def test_MedianAmplitude_extract():
    extractor = MedianAmplitude()

    # simulate results
    lc = {"magnitude": np.arange(LC_LENGTH + 1)}
    results = extractor.extract(**lc)

    # transform results into array
    values = np.array(list(results.values()))

    # assert mean is close to expected value
    expected = [475.0]  # MedianAmplitude
    np.testing.assert_allclose(values, expected)

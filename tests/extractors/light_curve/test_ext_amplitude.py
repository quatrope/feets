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

from feets.extractors.light_curve.ext_amplitude import Amplitude

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
def test_Amplitude_extract():
    # init extractor
    extractor = Amplitude()

    # simulate results
    lc = {"magnitude": np.arange(LC_LENGTH + 1)}
    kwargs = extractor.prepare_extract(lc, {})
    results = extractor.extract(**kwargs)

    # transform results into array
    values = np.array(list(results.values()))

    # assert results are close to expected value
    expected = [500]  # Amplitude
    np.testing.assert_allclose(values, expected)

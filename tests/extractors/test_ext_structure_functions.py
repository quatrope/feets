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

from feets.extractors.ext_structure_functions import StructureFunctions

import numpy as np

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
MAX_ITERS = 1000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_StructureFunctions_extract(normal):
    # init extractor
    extractor = StructureFunctions()

    # seed
    random = np.random.default_rng(RANDOM_SEED)

    # simulate results
    lcs = [
        {
            "time": np.arange(LC_LENGTH),
            "magnitude": normal(random=random, size=LC_LENGTH),
        }
        for _ in range(MAX_ITERS)
    ]
    results = [extractor.extract(**lc) for lc in lcs]

    # values by feature
    df = pd.DataFrame(results)

    # expected columns and mean values
    expected = pd.Series(
        {
            "StructureFunction_index_21": 1.8438983006429244,
            "StructureFunction_index_31": 2.637129298119476,
            "StructureFunction_index_32": 1.525586514233299,
        }
    )

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected, rtol=1e-2)


def test_StructureFunctions_extract_zeros():
    # init extractor
    extractor = StructureFunctions()

    # simulate results
    lc = {
        "time": np.arange(LC_LENGTH),
        "magnitude": np.zeros(LC_LENGTH),
    }
    result = extractor.extract(**lc)

    expected = {
        "StructureFunction_index_21": np.nan,
        "StructureFunction_index_31": np.nan,
        "StructureFunction_index_32": np.nan,
    }

    # check values
    np.testing.assert_equal(result, expected)

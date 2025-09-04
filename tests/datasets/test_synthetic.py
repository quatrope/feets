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

from feets.datasets.synthetic import (
    create_normal,
    create_periodic,
    create_uniform,
)

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET_SIZE = 10000
RANDOM_SEED = 42

# =============================================================================
# TESTS
# =============================================================================


def test_normal():
    np.random.seed(RANDOM_SEED)

    magnitude = np.random.normal(size=DATASET_SIZE)
    error = np.random.normal(size=DATASET_SIZE)

    ds = create_normal(seed=RANDOM_SEED, bands=["N"])

    np.testing.assert_equal(magnitude, ds.data.N.magnitude)
    np.testing.assert_equal(error, ds.data.N.error)


def test_uniform():
    np.random.seed(RANDOM_SEED)

    magnitude = np.random.uniform(size=DATASET_SIZE)
    error = np.random.normal(size=DATASET_SIZE)

    ds = create_uniform(seed=RANDOM_SEED, bands=["N"])

    np.testing.assert_equal(magnitude, ds.data.N.magnitude)
    np.testing.assert_equal(error, ds.data.N.error)


def test_periodic():
    np.random.seed(42)

    time = 100 * np.random.rand(DATASET_SIZE)
    error = np.random.normal(size=DATASET_SIZE)
    magnitude = np.sin(2 * np.pi * time) + error * np.random.randn(
        DATASET_SIZE
    )

    ds = create_periodic(seed=42, bands=["N"])

    np.testing.assert_equal(time, ds.data.N.time)
    np.testing.assert_equal(magnitude, ds.data.N.magnitude)
    np.testing.assert_equal(error, ds.data.N.error)

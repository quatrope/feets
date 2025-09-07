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

from feets.extractors.ext_slotted_a_length import (
    SlottedALength,
    slotted_autocorrelation,
    start_conditions,
)

import numpy as np

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

LC_LENGTH = 1000
MAX_ITERS = 100
RANDOM_SEED = 42

# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_slotted_autocorrelation():
    period = 20
    time = np.arange(LC_LENGTH)
    magnitude = np.sin(2 * np.pi * time / period)

    T = 1
    K = 100

    prod, slots = slotted_autocorrelation(magnitude, time, T, K)

    np.testing.assert_equal(prod.shape, (K, 1))

    np.testing.assert_allclose(prod[::20], 1, atol=1e-2)
    np.testing.assert_allclose(prod[5::20], 0, atol=1e-2)
    np.testing.assert_allclose(prod[10::20], -1, atol=1e-2)
    np.testing.assert_allclose(prod[15::20], 0, atol=1e-2)

    np.testing.assert_array_equal(slots, np.arange(K))


def test_slotted_autocorrelation_second_round():
    period = 20
    time = np.arange(LC_LENGTH)
    magnitude = np.sin(2 * np.pi * time / period)

    T = 1
    K = 150
    K1 = 100

    prod, slots = slotted_autocorrelation(
        magnitude, time, T, K, second_round=True, K1=K1
    )

    np.testing.assert_equal(prod.shape, (K, 1))

    np.testing.assert_allclose(prod[0], 1)
    np.testing.assert_allclose(prod[1:K1], 0)
    np.testing.assert_allclose(prod[-50::20], 1, atol=1e-2)
    np.testing.assert_allclose(prod[5 - 50 :: 20], 0, atol=1e-2)
    np.testing.assert_allclose(prod[10 - 50 :: 20], -1, atol=1e-2)
    np.testing.assert_allclose(prod[15 - 50 :: 20], 0, atol=1e-2)

    np.testing.assert_array_equal(slots, np.arange(K1, K))


def test_slotted_autocorrelation_second_round_no_trim():
    period = 20
    time = np.arange(10)
    magnitude = np.sin(2 * np.pi * time / period)

    T = 1
    K1 = 5
    K = 15

    prod, slots = slotted_autocorrelation(
        magnitude, time, T, K, second_round=True, K1=K1
    )

    np.testing.assert_equal(prod.shape, (K, 1))

    expected_zero_indices = np.arange(1, K1)
    expected_inf_indices = np.arange(10, K)
    expected_slots = np.arange(K1, 10)

    np.testing.assert_allclose(prod[expected_zero_indices], 0)
    np.testing.assert_allclose(prod[expected_inf_indices], np.inf)
    np.testing.assert_array_equal(slots, expected_slots)


def test_slotted_autocorrelation_no_pairs():
    period = 20
    time = np.concatenate([np.arange(10), np.arange(100, 110)])
    magnitude = np.sin(2 * np.pi * time / period)

    T = 1
    K = 50

    prod, slots = slotted_autocorrelation(magnitude, time, T, K)

    expected_slots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    np.testing.assert_array_equal(slots, expected_slots)

    assert not np.any(np.isinf(prod[slots]))

    inf_indices = np.setdiff1d(np.arange(1, K), slots)
    assert np.all(np.isinf(prod[inf_indices]))


def test_start_conditions():
    period = 20
    time = np.arange(LC_LENGTH)
    magnitude = np.sin(2 * np.pi * time / period)
    T_in = 1

    T_out, K, slots, SAC2 = start_conditions(magnitude, time, T=T_in)

    np.testing.assert_equal(T_out, T_in)
    np.testing.assert_equal(K, 100)

    np.testing.assert_equal(SAC2.shape, (K, 1))

    np.testing.assert_allclose(SAC2[::20], 1, atol=1e-2)
    np.testing.assert_allclose(SAC2[5::20], 0, atol=1e-2)
    np.testing.assert_allclose(SAC2[10::20], -1, atol=1e-2)
    np.testing.assert_allclose(SAC2[15::20], 0, atol=1e-2)

    np.testing.assert_array_equal(slots, np.arange(K))


def test_start_conditions_default_T():
    period = 20
    time = np.arange(LC_LENGTH)
    magnitude = np.sin(2 * np.pi * time / period)

    T, K, slots, SAC2 = start_conditions(magnitude, time)

    np.testing.assert_allclose(T, 1)
    np.testing.assert_allclose(K, 100)

    np.testing.assert_equal(SAC2.shape, (K, 1))

    np.testing.assert_allclose(SAC2[::20], 1, atol=1e-2)
    np.testing.assert_allclose(SAC2[5::20], 0, atol=1e-2)
    np.testing.assert_allclose(SAC2[10::20], -1, atol=1e-2)
    np.testing.assert_allclose(SAC2[15::20], 0, atol=1e-2)

    np.testing.assert_array_equal(slots, np.arange(K))


# =============================================================================
# TEST EXTRACTOR
# =============================================================================


def test_SlottedALength_extract(normal):
    # init extractor
    extractor = SlottedALength()

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
    expected = pd.Series({"SlottedALength": 1.0})

    # check columns
    np.testing.assert_equal(set(df.columns), set(expected.index))

    # check means
    means = df.mean()[expected.index]
    np.testing.assert_allclose(means, expected)


def test_SlottedALength_extract_k_doubling():
    time = np.arange(LC_LENGTH)
    # Decay constant chosen to require at least one K doubling.
    magnitude = np.exp(-0.008 * time)

    extractor = SlottedALength(T=1)

    result = extractor.extract(magnitude=magnitude, time=time)

    expected = {"SlottedALength": 127}

    np.testing.assert_equal(result, expected)


def test_SlottedALength_extract_nan():
    time = np.arange(LC_LENGTH)
    magnitude = np.ones(LC_LENGTH)

    extractor = SlottedALength(T=1)

    result = extractor.extract(magnitude=magnitude, time=time)

    expected = {"SlottedALength": np.nan}

    np.testing.assert_equal(result, expected)

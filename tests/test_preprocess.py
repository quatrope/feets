#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.preprocess import align, remove_noise

import numpy as np

# =============================================================================
# TESTS
# =============================================================================


def test_remove_noise_no_noise():
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    magnitude = [9.6, 9.7, 9.8, 9.9, 10, 10.1, 10.2, 10.3, 10.4, 10.5]
    error = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, time)
    np.testing.assert_array_equal(magnitude_clean, magnitude)
    np.testing.assert_array_equal(error_clean, error)


def test_remove_noise_with_error_noise():
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    magnitude = [9.6, 9.7, 9.8, 9.9, 10, 10.1, 10.2, 10.3, 10.4, 10.5]
    error = [0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, [1, 2, 3, 5, 6, 7, 8, 9, 10])
    np.testing.assert_array_equal(
        magnitude_clean, [9.6, 9.7, 9.8, 10, 10.1, 10.2, 10.3, 10.4, 10.5]
    )
    np.testing.assert_array_equal(
        error_clean, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )


def test_remove_noise_with_std_noise():
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    magnitude = [9.6, 9.7, 9.8, 9.9, 10, 10.1, 1000, 10.3, 10.4, 10.5]

    error = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, [1, 2, 3, 4, 5, 6, 8, 9, 10])
    np.testing.assert_array_equal(
        magnitude_clean, [9.6, 9.7, 9.8, 9.9, 10, 10.1, 10.3, 10.4, 10.5]
    )
    np.testing.assert_array_equal(
        error_clean, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )


def test_remove_noise_with_combined_noise():
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    magnitude = [9.6, 9.7, 9.8, 9.9, 10, 10.1, 1000, 10.3, 10.4, 10.5]

    error = [0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, [1, 2, 3, 5, 6, 8, 9, 10])
    np.testing.assert_array_equal(
        magnitude_clean, [9.6, 9.7, 9.8, 10, 10.1, 10.3, 10.4, 10.5]
    )
    np.testing.assert_array_equal(
        error_clean, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )


def test_remove_noise_empty_input():
    time, magnitude, error = [], [], []

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, [])
    np.testing.assert_array_equal(magnitude_clean, [])
    np.testing.assert_array_equal(error_clean, [])


def test_remove_noise_single_point():
    time, magnitude, error = np.array([1]), np.array([10]), np.array([0.1])

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, time)
    np.testing.assert_array_equal(magnitude_clean, magnitude)
    np.testing.assert_array_equal(error_clean, error)


def test_remove_noise_zero_error_mean():
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    magnitude = [9.6, 9.7, 9.8, 9.9, 10, 10.1, 10.2, 10.3, 10.4, 10.5]
    error = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    time_clean, magnitude_clean, error_clean = remove_noise(
        time, magnitude, error, error_limit=3, std_limit=2
    )

    np.testing.assert_array_equal(time_clean, time)
    np.testing.assert_array_equal(magnitude_clean, magnitude)
    np.testing.assert_array_equal(error_clean, error)


def test_align():
    time = [1, 2, 3, 4]
    magnitude = [10, 20, 30, 40]
    error = [0.1, 0.2, 0.3, 0.4]

    time2 = [1, 2, 4, 5, 6]
    magnitude2 = [11, 22, 44, 55, 66]
    error2 = [0.11, 0.22, 0.44, 0.55, 0.66]

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(time, time2, magnitude, magnitude2, error, error2)

    np.testing.assert_array_equal(aligned_time, [1, 2, 4])
    np.testing.assert_array_equal(aligned_magnitude, [10, 20, 40])
    np.testing.assert_array_equal(aligned_error, [0.1, 0.2, 0.4])
    np.testing.assert_array_equal(aligned_magnitude2, [11, 22, 44])
    np.testing.assert_array_equal(aligned_error2, [0.11, 0.22, 0.44])


def test_align_reversed_length():
    time = [1, 2, 4, 5, 6]
    magnitude = [11, 22, 44, 55, 66]
    error = [0.11, 0.22, 0.44, 0.55, 0.66]

    time2 = [1, 2, 3, 4]
    magnitude2 = [10, 20, 30, 40]
    error2 = [0.1, 0.2, 0.3, 0.4]

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(time, time2, magnitude, magnitude2, error, error2)

    np.testing.assert_array_equal(aligned_time, [1, 2, 4])
    np.testing.assert_array_equal(aligned_magnitude, [11, 22, 44])
    np.testing.assert_array_equal(aligned_error, [0.11, 0.22, 0.44])
    np.testing.assert_array_equal(aligned_magnitude2, [10, 20, 40])
    np.testing.assert_array_equal(aligned_error2, [0.1, 0.2, 0.4])


def test_align_no_overlap():
    time = [1, 2, 3]
    magnitude = [10, 20, 30]
    error = [0.1, 0.2, 0.3]

    time2 = [4, 5, 6]
    magnitude2 = [40, 50, 60]
    error2 = [0.4, 0.5, 0.6]

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(time, time2, magnitude, magnitude2, error, error2)

    np.testing.assert_array_equal(aligned_time, [])
    np.testing.assert_array_equal(aligned_magnitude, [])
    np.testing.assert_array_equal(aligned_error, [])
    np.testing.assert_array_equal(aligned_magnitude2, [])
    np.testing.assert_array_equal(aligned_error2, [])


def test_align_full_overlap():
    time = [1, 2, 3]
    magnitude = [10, 20, 30]
    error = [0.1, 0.2, 0.3]

    time2 = [1, 2, 3]
    magnitude2 = [11, 22, 33]
    error2 = [0.11, 0.22, 0.33]

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(time, time2, magnitude, magnitude2, error, error2)

    np.testing.assert_array_equal(aligned_time, [1, 2, 3])
    np.testing.assert_array_equal(aligned_magnitude, [10, 20, 30])
    np.testing.assert_array_equal(aligned_error, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(aligned_magnitude2, [11, 22, 33])
    np.testing.assert_array_equal(aligned_error2, [0.11, 0.22, 0.33])


def test_align_empty():
    time = []
    magnitude = []
    error = []

    time2 = [1, 2, 3]
    magnitude2 = [10, 20, 30]
    error2 = [0.1, 0.2, 0.3]

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(time, time2, magnitude, magnitude2, error, error2)

    np.testing.assert_array_equal(aligned_time, [])
    np.testing.assert_array_equal(aligned_magnitude, [])
    np.testing.assert_array_equal(aligned_error, [])
    np.testing.assert_array_equal(aligned_magnitude2, [])
    np.testing.assert_array_equal(aligned_error2, [])


def test_align_no_error():
    time = [1, 2, 3, 4]
    magnitude = [10, 20, 30, 40]

    time2 = [1, 2, 4, 5, 6]
    magnitude2 = [11, 22, 44, 55, 66]

    (
        aligned_time,
        aligned_magnitude,
        aligned_magnitude2,
        aligned_error,
        aligned_error2,
    ) = align(time, time2, magnitude, magnitude2)

    np.testing.assert_array_equal(aligned_time, [1, 2, 4])
    np.testing.assert_array_equal(aligned_magnitude, [10, 20, 40])
    np.testing.assert_array_equal(aligned_error, [0, 0, 0])
    np.testing.assert_array_equal(aligned_magnitude2, [11, 22, 44])
    np.testing.assert_array_equal(aligned_error2, [0, 0, 0])

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# DOC
# =============================================================================

"""All feets preprocess tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from feets import preprocess


# =============================================================================
# NOISE
# =============================================================================


def test_remove_noise():
    random = np.random.RandomState(42)

    time = np.arange(5)

    mag = random.rand(5)
    mag[-1] = np.mean(mag) + 100 * np.std(mag)

    error = np.zeros(5)
    error[-1] = 10

    ptime, pmag, perror = preprocess.remove_noise(time, mag, error)

    assert len(ptime) == len(time) - 1
    assert len(pmag) == len(mag) - 1
    assert len(perror) == len(error) - 1

    np.testing.assert_array_equal(ptime, time[:-1])
    np.testing.assert_array_equal(pmag, mag[:-1])
    np.testing.assert_array_equal(perror, error[:-1])


def test_remove_noise_low_error():
    random = np.random.RandomState(42)

    time = np.arange(5)

    mag = random.rand(5)
    mag[-1] = np.mean(mag) + 100 * np.std(mag)

    error = np.zeros(5)

    ptime, pmag, perror = preprocess.remove_noise(time, mag, error)

    assert len(ptime) == len(time)
    assert len(pmag) == len(mag)
    assert len(perror) == len(error)

    np.testing.assert_array_equal(ptime, time)
    np.testing.assert_array_equal(pmag, mag)
    np.testing.assert_array_equal(perror, error)


def test_remove_noise_no_outlier():
    random = np.random.RandomState(42)

    time = np.arange(5)
    mag = random.rand(5)
    error = np.zeros(5)

    ptime, pmag, perror = preprocess.remove_noise(time, mag, error)

    assert len(ptime) == len(time)
    assert len(pmag) == len(mag)
    assert len(perror) == len(error)

    np.testing.assert_array_equal(ptime, time)
    np.testing.assert_array_equal(pmag, mag)
    np.testing.assert_array_equal(perror, error)


# =============================================================================
# ALIGN
# =============================================================================


def test_align():
    random = np.random.RandomState(42)

    time = np.arange(5)
    mag = random.rand(5)
    error = random.rand(5)

    time2 = np.arange(5)
    random.shuffle(time2)

    mag2 = mag[time2]
    error2 = error[time2]

    atime, amag, amag2, aerror, aerror2 = preprocess.align(
        time, time2, mag, mag2, error, error2
    )

    np.testing.assert_array_equal(amag, amag2)
    assert np.array_equal(amag, mag) or np.array_equal(amag, mag2)
    assert np.array_equal(amag2, mag) or np.array_equal(amag2, mag2)

    np.testing.assert_array_equal(aerror, aerror2)
    assert np.array_equal(aerror, error) or np.array_equal(aerror, error2)
    assert np.array_equal(aerror2, error) or np.array_equal(aerror2, error2)


def test_align_different_len():
    random = np.random.RandomState(42)

    time = np.arange(5)
    mag = random.rand(5)
    error = random.rand(5)

    time2 = np.arange(6)
    random.shuffle(time2)

    mag2 = np.hstack((mag, random.rand(1)))[time2]
    error2 = np.hstack((error, random.rand(1)))[time2]

    atime, amag, amag2, aerror, aerror2 = preprocess.align(
        time, time2, mag, mag2, error, error2
    )

    np.testing.assert_array_equal(amag, amag2)
    np.testing.assert_array_equal(aerror, aerror2)

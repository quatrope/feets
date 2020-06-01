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

"""All syntethic tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from feets.datasets import synthetic as syn


# =============================================================================
# BASE CLASS
# =============================================================================


def test_normal():
    random = np.random.RandomState(42)

    mag = random.normal(size=10000)
    error = random.normal(size=10000)

    ds = syn.create_normal(seed=42, bands=["N"])

    np.testing.assert_array_equal(mag, ds.data.N.magnitude)
    np.testing.assert_array_equal(error, ds.data.N.error)


def test_uniform():
    random = np.random.RandomState(42)

    mag = random.uniform(size=10000)
    error = random.normal(size=10000)

    ds = syn.create_uniform(seed=42, bands=["U"])

    np.testing.assert_array_equal(mag, ds.data.U.magnitude)
    np.testing.assert_array_equal(error, ds.data.U.error)


def test_periodic():
    random = np.random.RandomState(42)

    time = 100 * random.rand(10000)
    error = random.normal(size=10000)
    mag = np.sin(2 * np.pi * time) + error * random.randn(10000)

    ds = syn.create_periodic(seed=42, bands=["P"])

    np.testing.assert_array_equal(time, ds.data.P.time)
    np.testing.assert_array_equal(mag, ds.data.P.magnitude)
    np.testing.assert_array_equal(error, ds.data.P.error)

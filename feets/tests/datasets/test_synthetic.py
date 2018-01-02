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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """All ogle3 access tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import tarfile

import numpy as np

import pandas as pd

import mock

from ...datasets import synthetic as syn

from ..core import FeetsTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class NormalTestCase(FeetsTestCase):

    def test_normal(self):
        np.random.seed(42)

        mag = np.random.normal(size=10000)
        error = np.random.normal(size=10000)

        ds = syn.create_normal(seed=42, bands=["N"])

        self.assertArrayEqual(mag, ds.data.N.magnitude)
        self.assertArrayEqual(error, ds.data.N.error)


class UniformTestCase(FeetsTestCase):

    def test_uniform(self):
        np.random.seed(42)

        mag = np.random.uniform(size=10000)
        error = np.random.normal(size=10000)

        ds = syn.create_uniform(seed=42, bands=["U"])

        self.assertArrayEqual(mag, ds.data.U.magnitude)
        self.assertArrayEqual(error, ds.data.U.error)


class PeriodicTestCase(FeetsTestCase):

    def test_periodic(self):
        np.random.seed(42)

        time = 100 * np.random.rand(10000)
        error = np.random.normal(size=10000)
        mag = np.sin(2 * np.pi * time) + error * np.random.randn(10000)

        ds = syn.create_periodic(seed=42, bands=["P"])

        self.assertArrayEqual(time, ds.data.P.time)
        self.assertArrayEqual(mag, ds.data.P.magnitude)
        self.assertArrayEqual(error, ds.data.P.error)

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

__doc__ = """All feets base tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest
import random
import os

import numpy as np
import numpy.testing as npt

import six
from six.moves import range


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


# =============================================================================
# BASE CLASS
# =============================================================================


class FeetsTestCase(unittest.TestCase):

    def assertAllClose(self, a, b, **kwargs):
        return npt.assert_allclose(a, b, **kwargs)

    def assertArrayEqual(self, a, b, **kwargs):
        return npt.assert_array_equal(a, b, **kwargs)

    def assertAll(self, arr, **kwargs):
        assert np.all(arr), "'{}' is not all True".format(arr)

    def rrange(self, a, b):
        return range(random.randint(a, b))

    if six.PY2:
        assertRaisesRegex = six.assertRaisesRegex
        assertCountEqual = six.assertCountEqual

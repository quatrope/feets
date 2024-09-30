#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


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

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

"""All feets base tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from feets import preprocess

from .core import FeetsTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class RemoveNoiseTestCase(FeetsTestCase):

    def test_remove_noise(self):
        time = np.arange(5)
        mag = np.random.rand(5)
        error = np.zeros(5)
        preprocess.remove_noise(time, mag, error)


class AlignTestCase(FeetsTestCase):

    def test_align(self):
        time = np.arange(5)
        mag = np.random.rand(5)
        error = np.random.rand(5)

        time2 = np.arange(5)
        np.random.shuffle(time2)

        mag2 = mag[time2]
        error2 = error[time2]

        atime, amag, amag2, aerror, aerror2 = preprocess.align(
            time, time2, mag, mag2, error, error2)

        self.assertArrayEqual(amag, amag2)
        self.assertTrue(
            np.array_equal(amag, mag) or np.array_equal(amag, mag2))
        self.assertTrue(
            np.array_equal(amag2, mag) or np.array_equal(amag2, mag2))

        self.assertArrayEqual(aerror, aerror2)
        self.assertTrue(
            np.array_equal(aerror, error) or np.array_equal(aerror, error2))
        self.assertTrue(
            np.array_equal(aerror2, error) or np.array_equal(aerror2, error2))

    def test_align_different_len(self):
        time = np.arange(5)
        mag = np.random.rand(5)
        error = np.random.rand(5)

        time2 = np.arange(6)
        np.random.shuffle(time2)

        mag2 = np.hstack((mag, np.random.rand(1)))[time2]
        error2 = np.hstack((error, np.random.rand(1)))[time2]

        atime, amag, amag2, aerror, aerror2 = preprocess.align(
            time, time2, mag, mag2, error, error2)

        self.assertArrayEqual(amag, amag2)
        self.assertArrayEqual(aerror, aerror2)

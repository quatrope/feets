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

import numpy as np

from .. import preprocess

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
            time, time2, mag, mag2, error, error2
        )

        self.assertArrayEqual(amag, amag2)
        self.assertTrue(
            np.array_equal(amag, mag) or np.array_equal(amag, mag2)
        )
        self.assertTrue(
            np.array_equal(amag2, mag) or np.array_equal(amag2, mag2)
        )

        self.assertArrayEqual(aerror, aerror2)
        self.assertTrue(
            np.array_equal(aerror, error) or np.array_equal(aerror, error2)
        )
        self.assertTrue(
            np.array_equal(aerror2, error) or np.array_equal(aerror2, error2)
        )

    def test_align_different_len(self):
        time = np.arange(5)
        mag = np.random.rand(5)
        error = np.random.rand(5)

        time2 = np.arange(6)
        np.random.shuffle(time2)

        mag2 = np.hstack((mag, np.random.rand(1)))[time2]
        error2 = np.hstack((error, np.random.rand(1)))[time2]

        atime, amag, amag2, aerror, aerror2 = preprocess.align(
            time, time2, mag, mag2, error, error2
        )

        self.assertArrayEqual(amag, amag2)
        self.assertArrayEqual(aerror, aerror2)

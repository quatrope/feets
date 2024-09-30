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

__doc__ = """All ogle3 access tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

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

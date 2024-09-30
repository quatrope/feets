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

import mock

from .. import (
    FeatureSpace,
    Extractor,
    register_extractor,
    ExtractorContractError,
)

from .core import FeetsTestCase


# =============================================================================
# BASE CLASS
# =============================================================================


class FeatureSpaceTestCase(FeetsTestCase):

    def test_extract(self):
        space = FeatureSpace(only=["Amplitude"])
        magnitude = np.array(
            [
                0.46057565,
                0.51372940,
                0.70136533,
                0.21454228,
                0.54792300,
                0.33433717,
                0.44879870,
                0.55571062,
                0.24388037,
                0.44793366,
                0.30175873,
                0.88326381,
                0.12208977,
                0.37088649,
                0.59457310,
                0.74705894,
                0.24551664,
                0.36009236,
                0.80661981,
                0.04961063,
                0.87747311,
                0.97388975,
                0.95775496,
                0.34195989,
                0.54201036,
                0.87854618,
                0.07388174,
                0.21543205,
                0.59295337,
                0.56771493,
            ]
        )

        features, values = space.extract(magnitude=magnitude)
        self.assertTrue(len(features) == 1 and features[0] == "Amplitude")
        self.assertAllClose(values[features == "Amplitude"], 0.45203809)

    @mock.patch("feets.extractors._extractors", {})
    def test_features_order(self):

        @register_extractor
        class ReturnSame(Extractor):
            data = ["magnitude"]
            features = ["Same"]

            def fit(self, magnitude):
                return {"Same": magnitude[0]}

        space = FeatureSpace(only=["Same"])

        for _ in range(200):
            data = np.unique(np.random.randint(1, 1000, 10))
            np.random.shuffle(data)

            features, values_col = space.extract(magnitude=data)
            self.assertArrayEqual(data[0], values_col)

    def test_features_kwargs(self):
        # ok
        FeatureSpace(only=["CAR_sigma"], CAR={"minimize_method": "powell"})

        # invalid parameter
        with self.assertRaises(ExtractorContractError):
            FeatureSpace(only=["CAR_sigma"], CAR={"o": 1})

        # invalid parameter with valid parameter
        with self.assertRaises(ExtractorContractError):
            FeatureSpace(
                only=["CAR_sigma"], CAR={"o": 1, "minimize_method": "powell"}
            )

    @mock.patch("feets.extractors._extractors", {})
    def test_remove_by_dependencies(self):
        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["test_a", "test_a2"]

            def fit(self, *args):
                pass

        @register_extractor
        class B1(Extractor):
            data = ["magnitude"]
            features = ["test_b1"]
            dependencies = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class C(Extractor):
            data = ["magnitude"]
            features = ["test_c"]

            def fit(self, *args):
                pass

        fs = FeatureSpace(exclude=["test_a"])
        self.assertCountEqual(fs.features_, ["test_c", "test_a2"])

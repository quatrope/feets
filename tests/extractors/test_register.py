#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

import feets.extractors.register
from feets.extractors.extractor import Extractor
from feets.extractors.register import (
    available_features,
    extractor_of,
    is_extractor_registered,
    is_feature_registered,
    is_instance_or_is_extractor,
    register_extractor,
    registered_extractors,
    sort_by_dependencies,
)


import numpy as np


class A(Extractor):
    features = ["test_a"]

    def extract(self):
        return {"test_a": None}


def test_is_instance_or_is_extractor(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    class A(Extractor):
        features = ["test_a"]

        def extract(self):
            return {"test_a": None}

    class B:
        pass

    np.testing.assert_(is_instance_or_is_extractor(A))
    np.testing.assert_(is_instance_or_is_extractor(A()))
    np.testing.assert_(not is_instance_or_is_extractor(B))
    np.testing.assert_(not is_instance_or_is_extractor(B()))


def test_register_extractor(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class A(Extractor):
        features = ["test_a"]

        def extract(self):
            return {"test_a": None}

    extractors = tuple(feets.extractors.register._extractors.values())
    np.testing.assert_equal(extractors, (A,))


def test_registered_extractors(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class A(Extractor):
        features = ["test_a"]

        def extract(self):
            return {"test_a": None}

    class B(Extractor):
        features = ["test_b"]

        def extract(self):
            return {"test_b": None}

    extractors = tuple(registered_extractors().values())
    np.testing.assert_equal(extractors, (A,))


def test_is_feature_registered(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class A(Extractor):
        features = ["test_a1", "test_a2"]

        def extract(self):
            return {
                "test_a1": None,
                "test_a2": None,
            }

    class B(Extractor):
        features = ["test_b1", "test_b2"]

        def extract(self):
            return {
                "test_b1": None,
                "test_b2": None,
            }

    for feature in A.get_features():
        np.testing.assert_(is_feature_registered(feature))
    for feature in B.get_features():
        np.testing.assert_(not is_feature_registered(feature))


def test_is_extractor_registered(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class A(Extractor):
        features = ["test_a"]

        def extract(self):
            return {"test_a": None}

    class B(Extractor):
        features = ["test_b"]

        def extract(self):
            return {"test_b": None}

    np.testing.assert_(is_extractor_registered(A))
    np.testing.assert_(not is_extractor_registered(B))


def test_available_features(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class B(Extractor):
        features = ["test_b2", "test_b1"]

        def extract(self):
            return {
                "test_b2": None,
                "test_b1": None,
            }

    @register_extractor
    class A(Extractor):
        features = ["test_a"]

        def extract(self):
            return {"test_a": None}

    @register_extractor
    class C(Extractor):
        features = ["test_c"]

        def extract(self):
            return {"test_c": None}

    sorted_features = ["test_a", "test_b1", "test_b2", "test_c"]
    np.testing.assert_array_equal(available_features(), sorted_features)


def test_extractor_of(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class A(Extractor):
        features = ["test_a1", "test_a2", "test_a3"]

        def extract(self):
            return {
                "test_a1": None,
                "test_a2": None,
                "test_a3": None,
            }

    @register_extractor
    class B(Extractor):
        features = [
            "test_b1",
            "test_b2",
        ]

        def extract(self):
            return {
                "test_b1": None,
                "test_b2": None,
            }

    for feature in A.get_features():
        np.testing.assert_equal(extractor_of(feature), A)
    for feature in B.get_features():
        np.testing.assert_equal(extractor_of(feature), B)


def test_sort_by_dependencies(mocker):
    mocker.patch("feets.extractors.register._extractors", {})

    @register_extractor
    class A(Extractor):
        features = ["test_a"]

        def extract(self, magnitude):
            return {"test_a": None}

    @register_extractor
    class B1(Extractor):
        features = ["test_b1"]

        def extract(self, magnitude, test_a):
            return {"test_b1": None}

    @register_extractor
    class B2(Extractor):
        features = ["test_b2"]

        def extract(self, magnitude, test_a):
            return {"test_b2": None}

    @register_extractor
    class C(Extractor):
        features = ["test_c"]

        def extract(self, magnitude, test_a, test_b1, test_b2):
            return {"test_c": None}

    a, b1, b2, c = A(), B1(), B2(), C()
    exts = [c, b1, a, b2]
    plan = sort_by_dependencies(exts)

    np.testing.assert_equal(len(plan), 4)
    for idx, ext in enumerate(plan):
        if idx == 0:
            np.testing.assert_equal(ext, a)
        elif idx in (1, 2):
            np.testing.assert_(np.isin(ext, (b1, b2)))
        elif idx == 3:
            np.testing.assert_equal(ext, c)

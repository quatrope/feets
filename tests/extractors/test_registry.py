#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


from feets.extractors.extractor import Extractor
from feets.extractors.registry import (
    DependencyNotFound,
    ExtractorRegistry,
    FeatureAlreadyRegistered,
    FeatureNotFound,
)

import numpy as np
from numpy.testing import assert_raises

import pytest


# Mock Extractor classes for testing
class MockExtractorA(Extractor):
    features = ["feature1"]

    @staticmethod
    def get_data():
        return {"valid_data1"}

    def extract(self):
        pass


class MockExtractorB1(Extractor):
    features = ["feature2"]

    @staticmethod
    def get_data():
        return {"valid_data1", "valid_data2"}

    def extract(self, feature1):
        pass


class MockExtractorB2(Extractor):
    features = ["feature3"]

    @staticmethod
    def get_data():
        return {"valid_data1", "valid_data2"}

    def extract(self, feature1):
        pass


class MockExtractorC(Extractor):
    features = ["feature4"]

    @staticmethod
    def get_data():
        return {"valid_data3"}

    def extract(self, feature2, feature3):
        pass


@pytest.fixture
def registry(mocker):
    mocker.patch(
        "feets.extractors.registry.DATAS",
        ("valid_data1", "valid_data2", "valid_data3"),
    )
    return ExtractorRegistry()


def test_validate_is_extractor_valid(registry):
    registry.validate_is_extractor(MockExtractorA)


def test_validate_is_extractor_invalid(registry):
    with assert_raises(TypeError):
        registry.validate_is_extractor(object)


def test_register_extractor_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    np.testing.assert_equal(registry._features, {"feature1"})
    np.testing.assert_equal(registry._extractors, {extractor})
    np.testing.assert_equal(
        registry._feature_extractors, {"feature1": extractor}
    )


def test_register_extractor_missing_dependencies(mocker, registry):
    extractor = MockExtractorA
    mocker.patch.object(
        extractor, "get_dependencies", result_value={"missing_feature"}
    )
    with assert_raises(DependencyNotFound):
        registry.register_extractor(extractor)


def test_register_extractor_feature_already_registered(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    with assert_raises(FeatureAlreadyRegistered):
        registry.register_extractor(extractor)


def test_unregister_extractor_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    registry.unregister_extractor(extractor)
    np.testing.assert_equal(registry._features, set())
    np.testing.assert_equal(registry._extractors, set())
    np.testing.assert_equal(registry._feature_extractors, {})


def test_unregister_extractor_nonexistent(registry):
    extractor = MockExtractorA
    with assert_raises(ValueError):
        registry.unregister_extractor(extractor)


def test_is_feature_registered_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    assert registry.is_feature_registered("feature1")


def test_is_feature_registered_invalid(registry):
    assert not registry.is_feature_registered("feature2")


def test_is_extractor_registered_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    assert registry.is_extractor_registered(extractor)


def test_is_extractor_registered_invalid(registry):
    extractor = MockExtractorA
    assert not registry.is_extractor_registered(extractor)


def test_extractor_of_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    np.testing.assert_equal(registry.extractor_of("feature1"), extractor)


def test_extractor_of_invalid(registry):
    with assert_raises(FeatureNotFound):
        registry.extractor_of("feature1")


def test_extractors_from_data_valid(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    result = registry.extractors_from_features({"feature1", "feature4"})
    np.testing.assert_equal(result, {extractorA, extractorC})

    result = registry.extractors_from_data({"valid_data1", "valid_data2"})
    np.testing.assert_equal(result, {extractorA, extractorB1, extractorB2})


def test_extractors_from_data_invalid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    with assert_raises(ValueError):
        registry.extractors_from_data({"invalid_data"})


def test_extractors_from_features_valid(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    result = registry.extractors_from_features({"feature1", "feature4"})
    np.testing.assert_equal(result, {extractorA, extractorC})


def test_extractors_from_features_invalid(registry):
    with assert_raises(FeatureNotFound):
        registry.extractors_from_features({"feature1"})


def test_sort_extractors_by_dependencies_valid(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    extractors = [extractorB1, extractorC, extractorA]
    result = registry.sort_extractors_by_dependencies(extractors)

    np.testing.assert_equal(result[0], extractorA)
    np.testing.assert_equal(result[3], extractorC)
    assert result[1] in {extractorB1, extractorB2}
    assert result[2] in {extractorB1, extractorB2}


def test_sort_extractors_by_dependencies_invalid(registry):
    with assert_raises(TypeError):
        registry.sort_extractors_by_dependencies([object])


def test_get_execution_plan_data(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    result = registry.get_execution_plan(
        data={"valid_data1"},
    )

    np.testing.assert_array_equal(result, [extractorA])


def test_get_execution_plan_only(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    result = registry.get_execution_plan(
        only={"feature2"},
    )

    np.testing.assert_array_equal(result, [extractorA, extractorB1])


def test_get_execution_plan_exclude(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    result = registry.get_execution_plan(
        exclude={"feature4"},
    )

    np.testing.assert_equal(result[0], extractorA)
    assert result[1] in {extractorB1, extractorB2}
    assert result[2] in {extractorB1, extractorB2}


def test_get_execution_plan_all(registry):
    extractorA = MockExtractorA
    extractorB1 = MockExtractorB1
    extractorB2 = MockExtractorB2
    extractorC = MockExtractorC

    registry.register_extractor(extractorA)
    registry.register_extractor(extractorB1)
    registry.register_extractor(extractorB2)
    registry.register_extractor(extractorC)

    result = registry.get_execution_plan()

    np.testing.assert_equal(result[0], extractorA)
    np.testing.assert_equal(result[3], extractorC)
    assert result[1] in {extractorB1, extractorB2}
    assert result[2] in {extractorB1, extractorB2}


def test_get_execution_plan_disjoint(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    with assert_raises(ValueError):
        registry.get_execution_plan(
            data={"valid_data"}, only={"feature1"}, exclude={"feature1"}
        )

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

import pytest


# Mock Extractor classes for testing
class MockExtractorA(Extractor):
    features = ["feature1"]

    def extract(self):
        return {"feature1": None}


class MockExtractorB(Extractor):
    features = ["feature2"]

    def extract(self):
        return {"feature2": None}


@pytest.fixture
def registry(mocker):
    mocker.patch("feets.extractors.registry.DATAS", ("valid_data",))
    return ExtractorRegistry()


def test_validate_is_extractor_valid(registry):
    registry.validate_is_extractor(MockExtractorA)


def test_validate_is_extractor_invalid(registry):
    with pytest.raises(TypeError):
        registry.validate_is_extractor(object)


def test_register_extractor_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    assert registry.is_extractor_registered(extractor)


def test_register_extractor_missing_dependencies(mocker, registry):
    extractor = MockExtractorA
    mocker.patch.object(
        extractor, "get_dependencies", result_value={"missing_feature"}
    )
    with pytest.raises(DependencyNotFound):
        registry.register_extractor(extractor)


def test_register_extractor_feature_already_registered(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    with pytest.raises(FeatureAlreadyRegistered):
        registry.register_extractor(extractor)


def test_unregister_extractor_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    registry.unregister_extractor(extractor)
    assert not registry.is_extractor_registered(extractor)


def test_unregister_extractor_nonexistent(registry):
    extractor = MockExtractorA
    with pytest.raises(ValueError):
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
    assert registry.extractor_of("feature1") == extractor


def test_extractor_of_invalid(registry):
    with pytest.raises(FeatureNotFound):
        registry.extractor_of("feature2")


def test_extractors_from_data_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    result = registry.extractors_from_data({"valid_data"})
    assert extractor in result


def test_extractors_from_data_invalid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    with pytest.raises(ValueError):
        registry.extractors_from_data({"invalid_data"})


def test_extractors_from_features_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    result = registry.extractors_from_features({"feature1"})
    assert extractor in result


def test_extractors_from_features_invalid(registry):
    with pytest.raises(FeatureNotFound):
        registry.extractors_from_features({"feature2"})


def test_sort_extractors_by_dependencies_valid(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    result = registry.sort_extractors_by_dependencies([extractor])
    assert extractor in result


def test_sort_extractors_by_dependencies_invalid(registry):
    with pytest.raises(TypeError):
        registry.sort_extractors_by_dependencies([object])


def test_get_execution_plan_valid(registry):
    extractor_A = MockExtractorA
    extractor_B = MockExtractorB

    registry.register_extractor(extractor_A)
    registry.register_extractor(extractor_B)

    result = registry.get_execution_plan(
        data={"valid_data"},
        only={"feature1"},
        exclude={"feature2"},
    )
    assert extractor_A in result


def test_get_execution_plan_disjoint(registry):
    extractor = MockExtractorA
    registry.register_extractor(extractor)
    with pytest.raises(ValueError):
        registry.get_execution_plan(
            data={"valid_data"}, only={"feature1"}, exclude={"feature1"}
        )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.extractor import Extractor, ExtractorBadDefinedError
from feets.extractors.registry import (
    EntityNotFoundError,
    ExtractorRegistry,
    RegistryConflictError,
    RegistryValidationError,
)

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_1 = "test_data_1"
DATA_2 = "test_data_2"
DATA_3 = "test_data_3"
DATA_4 = "test_data_4"
ALL_DATA = (DATA_1, DATA_2, DATA_3, DATA_4)
INVALID_DATA = "test_data_5"

FEATURE_1 = "test_feature_1"
FEATURE_2 = "test_feature_2"
FEATURE_3 = "test_feature_3"
FEATURE_4 = "test_feature_4"

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def empty_registry(mocker):
    mocker.patch("feets.extractors.registry.DATAS", ALL_DATA)
    return ExtractorRegistry()


@pytest.fixture
def extractor_DATAS_mock(mocker):
    return mocker.patch("feets.extractors.extractor.DATAS", ALL_DATA)


@pytest.fixture
def TestExtractor1(extractor_DATAS_mock):
    class TestExtractor1(Extractor):
        features = {FEATURE_1}

        def extract(self, test_data_1, test_data_2, test_data_3, test_data_4):
            pass

    return TestExtractor1


@pytest.fixture
def TestExtractor2(extractor_DATAS_mock):

    class TestExtractor2(Extractor):
        features = {FEATURE_2}

        def extract(self, test_data_2, test_feature_1):
            pass

    return TestExtractor2


@pytest.fixture
def TestExtractor3(extractor_DATAS_mock):

    class TestExtractor3(Extractor):
        features = {FEATURE_3}

        def extract(self, test_data_3, test_feature_1):
            pass

    return TestExtractor3


@pytest.fixture
def TestExtractor4(extractor_DATAS_mock):

    class TestExtractor4(Extractor):
        features = {FEATURE_4}

        def extract(self, test_data_4, test_feature_2, test_feature_3):
            pass

    return TestExtractor4


@pytest.fixture
def TestConflictiveExtractor(extractor_DATAS_mock):

    class TestConflictiveExtractor(Extractor):
        features = {FEATURE_1}

        def extract(self):
            pass

    return TestConflictiveExtractor


@pytest.fixture
def registry(
    mocker, TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4
):
    mocker.patch("feets.extractors.registry.DATAS", ALL_DATA)
    reg = ExtractorRegistry()
    reg.register_extractor(TestExtractor1)
    reg.register_extractor(TestExtractor2)
    reg.register_extractor(TestExtractor3)
    reg.register_extractor(TestExtractor4)
    return reg


# =============================================================================
# TESTS
# =============================================================================


def test_ExtractorRegistry_validate_is_extractor_no_extract_method():
    class BadExtractor(Extractor):
        features = [FEATURE_1]

    with pytest.raises(ExtractorBadDefinedError):
        ExtractorRegistry.validate_is_extractor(BadExtractor)


def test_ExtractorRegistry_validate_is_extractor_abstract_subclass():
    class AbstractExtractor(Extractor):
        pass

    with pytest.raises(TypeError):
        ExtractorRegistry.validate_is_extractor(AbstractExtractor)


def test_ExtractorRegistry_validate_is_extractor_not_an_extractor():
    class NotAnExtractor:
        pass

    with pytest.raises(TypeError):
        ExtractorRegistry.validate_is_extractor(NotAnExtractor)


def test_ExtractorRegistry_register_extractor(
    empty_registry, TestExtractor1, TestExtractor2
):
    empty_registry.register_extractor(TestExtractor1)
    empty_registry.register_extractor(TestExtractor2)


def test_ExtractorRegistry_register_extractor_missing_dependencies(
    empty_registry, TestExtractor2
):
    with pytest.raises(EntityNotFoundError):
        empty_registry.register_extractor(TestExtractor2)


def test_ExtractorRegistry_register_extractor_feature_conflict(
    empty_registry, TestExtractor1, TestConflictiveExtractor
):
    empty_registry.register_extractor(TestExtractor1)
    with pytest.raises(RegistryConflictError):
        empty_registry.register_extractor(TestConflictiveExtractor)


def test_ExtractorRegistry_unregister_extractor(
    empty_registry, TestExtractor1, TestExtractor2
):
    empty_registry.register_extractor(TestExtractor1)
    empty_registry.register_extractor(TestExtractor2)

    empty_registry.unregister_extractor(TestExtractor2)
    empty_registry.unregister_extractor(TestExtractor1)


def test_ExtractorRegistry_unregister_extractor_missing_extractor(
    empty_registry, TestExtractor1
):
    with pytest.raises(EntityNotFoundError):
        empty_registry.unregister_extractor(TestExtractor1)


def test_ExtractorRegistry_unregister_extractor_dependency_conflict(
    empty_registry, TestExtractor1, TestExtractor2
):
    empty_registry.register_extractor(TestExtractor1)
    empty_registry.register_extractor(TestExtractor2)
    with pytest.raises(RegistryConflictError):
        empty_registry.unregister_extractor(TestExtractor1)


def test_ExtractorRegistry_is_feature_registered(
    empty_registry, TestExtractor1, TestExtractor2
):
    assert not empty_registry.is_feature_registered(FEATURE_1)
    assert not empty_registry.is_feature_registered(FEATURE_2)

    empty_registry.register_extractor(TestExtractor1)
    assert empty_registry.is_feature_registered(FEATURE_1)
    assert not empty_registry.is_feature_registered(FEATURE_2)

    empty_registry.register_extractor(TestExtractor2)
    assert empty_registry.is_feature_registered(FEATURE_1)
    assert empty_registry.is_feature_registered(FEATURE_2)

    empty_registry.unregister_extractor(TestExtractor2)
    assert empty_registry.is_feature_registered(FEATURE_1)
    assert not empty_registry.is_feature_registered(FEATURE_2)

    empty_registry.unregister_extractor(TestExtractor1)
    assert not empty_registry.is_feature_registered(FEATURE_1)
    assert not empty_registry.is_feature_registered(FEATURE_2)


def test_ExtractorRegistry_is_extractor_registered(
    empty_registry, TestExtractor1, TestExtractor2
):
    assert not empty_registry.is_extractor_registered(TestExtractor1)
    assert not empty_registry.is_extractor_registered(TestExtractor2)

    empty_registry.register_extractor(TestExtractor1)
    assert empty_registry.is_extractor_registered(TestExtractor1)
    assert not empty_registry.is_extractor_registered(TestExtractor2)

    empty_registry.register_extractor(TestExtractor2)
    assert empty_registry.is_extractor_registered(TestExtractor1)
    assert empty_registry.is_extractor_registered(TestExtractor2)

    empty_registry.unregister_extractor(TestExtractor2)
    assert empty_registry.is_extractor_registered(TestExtractor1)
    assert not empty_registry.is_extractor_registered(TestExtractor2)

    empty_registry.unregister_extractor(TestExtractor1)
    assert not empty_registry.is_extractor_registered(TestExtractor1)
    assert not empty_registry.is_extractor_registered(TestExtractor2)


def test_ExtractorRegistry_extractor_of(empty_registry, TestExtractor1):
    empty_registry.register_extractor(TestExtractor1)
    np.testing.assert_equal(
        empty_registry.extractor_of(FEATURE_1), TestExtractor1
    )


def test_ExtractorRegistry_extractor_of_missing_feature(empty_registry):
    with pytest.raises(EntityNotFoundError):
        empty_registry.extractor_of(FEATURE_1)


pytest.mark.parametrize(
    ["data", "expected_extractors"],
    [
        ([DATA_1], set()),
        ([DATA_2], {TestExtractor2}),
        ([DATA_3, DATA_4], {TestExtractor3, TestExtractor4}),
        (
            [DATA_1, DATA_2, DATA_3, DATA_4],
            {TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4},
        ),
    ],
)


def test_ExtractorRegistry_extractors_from_data(
    registry,
    TestExtractor1,
    TestExtractor2,
    TestExtractor3,
    TestExtractor4,
):

    np.testing.assert_equal(registry.extractors_from_data([DATA_1]), set())
    np.testing.assert_equal(
        registry.extractors_from_data([DATA_2]), {TestExtractor2}
    )
    np.testing.assert_equal(
        registry.extractors_from_data([DATA_3, DATA_4]),
        {TestExtractor3, TestExtractor4},
    )
    np.testing.assert_equal(
        registry.extractors_from_data([DATA_1, DATA_2, DATA_3, DATA_4]),
        {TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4},
    )


def test_ExtractorRegistry_extractors_from_data_invalid_data(
    empty_registry, TestExtractor1
):
    empty_registry.register_extractor(TestExtractor1)
    with pytest.raises(RegistryValidationError):
        empty_registry.extractors_from_data([INVALID_DATA])


def test_ExtractorRegistry_extractors_from_features(
    registry,
    TestExtractor1,
    TestExtractor2,
    TestExtractor3,
    TestExtractor4,
):
    np.testing.assert_equal(
        registry.extractors_from_features([FEATURE_1]), {TestExtractor1}
    )
    np.testing.assert_equal(
        registry.extractors_from_features([FEATURE_2]), {TestExtractor2}
    )
    np.testing.assert_equal(
        registry.extractors_from_features([FEATURE_3, FEATURE_4]),
        {TestExtractor3, TestExtractor4},
    )
    np.testing.assert_equal(
        registry.extractors_from_features(
            [FEATURE_1, FEATURE_2, FEATURE_3, FEATURE_4]
        ),
        {TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4},
    )


def test_ExtractorRegistry_extractors_from_features_missing_feature(
    empty_registry, TestExtractor1
):
    empty_registry.register_extractor(TestExtractor1)
    with pytest.raises(EntityNotFoundError):
        empty_registry.extractors_from_features([FEATURE_2])


def test_ExtractorRegistry_sort_extractors_by_dependencies(
    registry,
    TestExtractor1,
    TestExtractor2,
    TestExtractor3,
    TestExtractor4,
):
    np.testing.assert_equal(registry.sort_extractors_by_dependencies([]), [])

    np.testing.assert_equal(
        registry.sort_extractors_by_dependencies(
            [TestExtractor2, TestExtractor1]
        ),
        (TestExtractor1, TestExtractor2),
    )

    result = registry.sort_extractors_by_dependencies([TestExtractor4])
    np.testing.assert_equal(
        (result[0], set(result[1:3]), result[3]),
        (TestExtractor1, {TestExtractor2, TestExtractor3}, TestExtractor4),
    )


def test_ExtractorRegistry_sort_extractors_by_dependencies_missing_extractor(
    empty_registry, TestExtractor1, TestExtractor2
):
    empty_registry.register_extractor(TestExtractor1)
    with pytest.raises(EntityNotFoundError):
        empty_registry.sort_extractors_by_dependencies([TestExtractor2])


def test_ExtractorRegistry_get_execution_plan(
    registry, TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4
):
    result = registry.get_execution_plan()
    np.testing.assert_equal(
        (result[0], set(result[1:3]), result[3]),
        (TestExtractor1, {TestExtractor2, TestExtractor3}, TestExtractor4),
    )


def test_ExtractorRegistry_get_execution_plan_filters(
    registry, TestExtractor1, TestExtractor2, TestExtractor3
):
    filters = {
        "data": [DATA_2, DATA_3],
        "only": [FEATURE_1, FEATURE_2],
    }
    np.testing.assert_equal(
        registry.get_execution_plan(**filters),
        (TestExtractor1, TestExtractor2),
    )

    filters = {
        "data": [DATA_2, DATA_3],
        "exclude": [FEATURE_2],
    }
    np.testing.assert_equal(
        registry.get_execution_plan(**filters),
        (TestExtractor1, TestExtractor3),
    )


def test_ExtractorRegistry_get_execution_plan_filters_invalid_data(
    empty_registry, TestExtractor1
):
    empty_registry.register_extractor(TestExtractor1)
    filters = {"data": [INVALID_DATA]}
    with pytest.raises(RegistryValidationError):
        empty_registry.get_execution_plan(**filters)


def test_ExtractorRegistry_get_execution_plan_filters_conflictive_filters(
    empty_registry, TestExtractor1
):
    empty_registry.register_extractor(TestExtractor1)
    filters = {"only": [FEATURE_1], "exclude": [FEATURE_1]}
    with pytest.raises(RegistryValidationError):
        empty_registry.get_execution_plan(**filters)


def test_ExtractorRegistry_get_execution_plan_filters_missing_only(
    empty_registry, TestExtractor1
):
    empty_registry.register_extractor(TestExtractor1)
    filters = {"only": [FEATURE_2]}
    with pytest.raises(EntityNotFoundError):
        empty_registry.get_execution_plan(**filters)


def test_ExtractorRegistry_get_execution_plan_filters_missing_exclude(
    empty_registry, TestExtractor1
):
    empty_registry.register_extractor(TestExtractor1)
    filters = {"exclude": [FEATURE_2]}
    with pytest.raises(EntityNotFoundError):
        empty_registry.get_execution_plan(**filters)


def test_ExtractorRegistry_registered_extractors(
    registry, TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4
):
    np.testing.assert_equal(
        registry.registered_extractors,
        {TestExtractor1, TestExtractor2, TestExtractor3, TestExtractor4},
    )


def test_ExtractorRegistry_registered_features(registry):
    np.testing.assert_equal(
        registry.registered_features,
        {FEATURE_1, FEATURE_2, FEATURE_3, FEATURE_4},
    )

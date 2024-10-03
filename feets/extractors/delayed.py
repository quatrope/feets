#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from .extractor import ExtractorContractError


# =============================================================================
# DELAYED EXTRACTOR
# =============================================================================


def select_extract_kwargs(extractor, data, features):
    kwargs = {required: data[required] for required in extractor.get_data()}

    for dependency in extractor.get_dependencies():
        kwargs[dependency] = features[dependency]

    return kwargs


def _validate_extracted_features(extractor, extracted_features):
    # validate if the extractor generates the expected features
    if extracted_features is None:
        extracted_features = {}

    expected_features = extractor.get_features()
    if expected_features is None:
        expected_features = frozenset()

    diff = set(extracted_features).symmetric_difference(expected_features)
    if diff:
        cls_name = type(extractor).__qualname__
        estr, fstr = ", ".join(expected_features), ", ".join(
            extracted_features.keys()
        )
        raise ExtractorContractError(
            f"The extractor '{cls_name}' expected the features {estr}. "
            f"Found: {fstr!r}"
        )

    return extracted_features


def validate_and_extract(extractor, kwargs):
    extracted_features = extractor.extract(**kwargs)
    _validate_extracted_features(extractor, extracted_features)
    return extracted_features


def select_feature(extraction, feature):
    return extraction[feature]

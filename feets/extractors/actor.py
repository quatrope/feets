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

import ray

from .extractor import ExtractorContractError


# =============================================================================
# EXTRACTOR ACTOR
# =============================================================================


@ray.remote
class ExtractorActor:
    def __init__(self, extractor, result_refs_by_dependency):
        self._extractor = extractor
        self._refs = result_refs_by_dependency

    def preprocess_arguments(self, data):
        kwargs = {}

        # add the required features
        for d in self._extractor.get_dependencies():
            result_ref = self._refs[d]
            result = ray.get(result_ref)
            kwargs[d] = result[d]

        # add the required data
        for d in self._extractor.get_data():
            kwargs[d] = data[d]

        return kwargs

    def validate_result(self, result):
        if result is None:
            result = dict()

        # validate if the extractor generates the expected features
        expected_features = self._extractor.get_features()
        if expected_features is None:
            expected_features = frozenset()

        diff = set(result).symmetric_difference(expected_features)
        if diff:
            cls_name = type(self).__qualname__
            estr, fstr = ", ".join(expected_features), ", ".join(result.keys())
            raise ExtractorContractError(
                f"The extractor '{cls_name}' expected the features {estr}. "
                f"Found: {fstr!r}"
            )

        return result

    def select_extract_and_validate(self, data):
        extract_kwargs = self.preprocess_arguments(data)

        results = self._extractor.extract(**extract_kwargs)

        self.validate_result(results)

        return results

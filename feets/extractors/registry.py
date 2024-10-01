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

from .extractor import (
    DATAS,
    Extractor,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class DependencyNotFound(ValueError):
    def __init__(self, dependencies) -> None:
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        dependencies_str = ", ".join(dependencies)
        super().__init__(f"Dependencies not found: {dependencies_str}")


class FeatureNotFound(ValueError):
    def __init__(self, features):
        if isinstance(features, str):
            features = [features]
        features_str = ", ".join(features)
        super().__init__(f"Features not found: {features_str}")


class FeatureAlreadyRegistered(ValueError):
    def __init__(self, features):
        if isinstance(features, str):
            features = [features]
        features_str = ", ".join(features)
        super().__init__(f"Features are already registered: {features_str}")


# =============================================================================
# EXTRACTOR REGISTRY CLASS
# =============================================================================


class ExtractorRegistry:
    def __init__(self):
        self._feature_extractors = {}
        self._features = set()
        self._extractors = set()

    def validate_is_extractor(self, cls):
        if not issubclass(cls, Extractor):
            raise TypeError(
                f"Only Extractor subclasses are allowed. Found: '{cls}'."
            )

    def register_extractor(self, cls):
        self.validate_is_extractor(cls)

        # check dependencies
        missing_dependencies = cls.get_dependencies().difference(
            self._features
        )
        if missing_dependencies:
            raise DependencyNotFound(missing_dependencies)

        # check if features are already registered
        registered_features = cls.get_features().intersection(self._features)
        if registered_features:
            raise FeatureAlreadyRegistered(registered_features)

        # register the extractor
        for feature in cls.get_features():
            self._feature_extractors[feature] = cls
            self._features.add(feature)
        self._extractors.add(cls)

        return cls

    def unregister_extractor(self, cls):
        self.validate_is_extractor(cls)

        # check if the extractor is registered
        if cls not in self._extractors:
            raise ValueError(f"Extractor '{cls}' is not registered.")

        features = cls.get_features()

        # check dependencies
        for extractor in self._extractors.difference([cls]):
            if features.intersection(extractor.get_dependencies()):
                raise ValueError(
                    f"Extractor '{cls}' is a dependency of extractor "
                    f"'{extractor}'."
                )

        # unregister extractor
        for feature in features:
            del self._feature_extractors[feature]
            self._features.remove(feature)
        self._extractors.remove(cls)

    def is_feature_registered(self, feature):
        return feature in self._features

    def is_extractor_registered(self, extractor):
        self.validate_is_extractor(extractor)

        return extractor in self._extractors

    def extractor_of(self, feature):
        if not self.is_feature_registered(feature):
            raise FeatureNotFound(feature)

        return self._feature_extractors[feature]

    def extractors_from_data(self, data):
        diff = set(data).difference(DATAS)
        if diff:
            raise ValueError(f"Invalid data: {', '.join(diff)}.")

        return {
            extractor
            for extractor in self._extractors
            if extractor.get_data().issubset(data)
        }

    def extractors_from_features(self, features):
        extractors = set()
        for feature in features:
            if not self.is_feature_registered(feature):
                raise FeatureNotFound(feature)
            extractors.add(self._feature_extractors[feature])
        return extractors

    def sort_extractors_by_dependencies(self, extractors):
        """Calculate the Feature Extractor Resolution Order."""
        selected_extractors = []
        features_from_selected = set()
        pending = list(extractors)
        processed = set()

        while pending:
            extractor = pending.pop(0)
            if not self.is_extractor_registered(extractor):
                raise ValueError(f"Extractor '{extractor}' is not registered.")
            if extractor in processed:
                continue

            missing_dependencies = extractor.get_dependencies().difference(
                features_from_selected
            )
            # If dependencies are not in the selected features, add them
            if missing_dependencies:
                pending.extend(
                    self._feature_extractors[dependency]
                    for dependency in missing_dependencies
                )
                # Re-add the current extractor to pending for another check
                pending.append(extractor)
            else:
                selected_extractors.append(extractor)
                features_from_selected.update(extractor.get_features())
                processed.add(extractor)

        return tuple(selected_extractors)

    def get_execution_plan(self, *, data=None, only=None, exclude=None):
        if not set(only or []).isdisjoint(exclude or []):
            raise ValueError(
                "Features in 'only' and 'exclude' must be disjoint."
            )

        from_data = (
            self.extractors_from_data(data)
            if data is not None
            else self._extractors
        )
        from_only = (
            self.extractors_from_features(only)
            if only is not None
            else self._extractors
        )
        from_exclude = (
            self.extractors_from_features(exclude)
            if exclude is not None
            else set({})
        )

        selected_extractors = (
            set(from_data).intersection(from_only).difference(from_exclude)
        )

        return self.sort_extractors_by_dependencies(selected_extractors)

    @property
    def registered_extractors(self):
        return frozenset(self._extractors)

    @property
    def available_features(self):
        return frozenset(self._features)

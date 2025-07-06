#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Register and manage feature extractors."""


# =============================================================================
# IMPORTS
# =============================================================================

from .light_curve.light_curve_extractor import LightCurveExtractor
from .extractor import DATAS, Extractor


# =============================================================================
# EXCEPTIONS
# =============================================================================


class RegistryError(Exception):
    """Base class for all registry-related errors."""

    pass


class EntityNotFoundError(RegistryError):
    """An extractor or feature is not present in the registry."""

    pass


class RegistryConflictError(RegistryError):
    """A conflict occurred during registration or unregistration."""

    pass


class RegistryValidationError(RegistryError):
    """An error occurred due to invalid parameters."""

    pass


# =============================================================================
# EXTRACTOR REGISTRY CLASS
# =============================================================================


class ExtractorRegistry:
    """Extractor registry for managing feature extractors.

    The `ExtractorRegistry` class is responsible for managing the registration
    and unregistration of feature extractors. It ensures that all dependencies
    are met before registering an extractor and prevents duplicate feature
    registrations.

    It also provides methods to check if a feature or extractor is registered,
    retrieve the extractor for a specific feature, and generate an execution
    plan for extractors based on provided data and feature constraints.
    """

    def __init__(self):
        self._feature_extractors = {}
        self._features = set()
        self._extractors = set()

    def validate_is_extractor(self, cls):
        """Validate that a class is a subclass of Extractor.

        Parameters
        ----------
        cls : class
            The class to validate.

        Raises
        ------
        TypeError
            If the class is not a subclass of Extractor.
        """
        if not issubclass(cls, Extractor) and not issubclass(
            cls, LightCurveExtractor
        ):
            raise TypeError(
                f"Only Extractor subclasses are allowed. Found: '{cls}'."
            )

    def register_extractor(self, cls):
        """Add a feature extractor to the registry.

        Ensure that all dependencies are met before registering the extractor.

        Parameters
        ----------
        cls : class
            The feature extractor class to register.

        Returns
        -------
        Extractor
            The registered feature extractor class.

        Raises
        ------
        EntityNotFoundError
            If one of the dependencies of the extractor is not present in the
            registry.
        RegistryConflictError
            If one of the features of the extractor is already registered.
        """
        self.validate_is_extractor(cls)

        # check dependencies
        missing_dependencies = cls.get_dependencies().difference(
            self._features
        )
        if missing_dependencies:
            raise EntityNotFoundError(
                f"Extractors not found: {', '.join(map(repr, missing_dependencies))}"
            )

        # check if features are already registered
        registered_features = cls.get_features().intersection(self._features)
        if registered_features:
            raise RegistryConflictError(
                f"Features already registered: {', '.join(map(repr, registered_features))}"
            )

        # register the extractor
        for feature in cls.get_features():
            self._feature_extractors[feature] = cls
            self._features.add(feature)
        self._extractors.add(cls)

        return cls

    def unregister_extractor(self, cls):
        """Remove a feature extractor from the registry.

        Parameters
        ----------
        cls : class
            The feature extractor class to unregister.

        Raises
        ------
        EntityNotFoundError
            If the extractor is not present in the registry

        RegistryConflictError
            If the extractor is a dependency of another extractor in the registry.
        """
        self.validate_is_extractor(cls)

        # check if the extractor is registered
        if cls not in self._extractors:
            raise EntityNotFoundError(f"Extractor '{cls}' not found.")

        features = cls.get_features()

        # check dependencies
        for extractor in self._extractors.difference([cls]):
            if features.intersection(extractor.get_dependencies()):
                raise RegistryConflictError(
                    f"Extractor '{cls}' is a dependency of extractor '{extractor}'."
                )

        # unregister extractor
        for feature in features:
            del self._feature_extractors[feature]
            self._features.remove(feature)
        self._extractors.remove(cls)

    def is_feature_registered(self, feature):
        """Check if a feature is extracted by any registered extractor.

        Parameters
        ----------
        feature : str
            The name of the feature to check.

        Returns
        -------
        bool
            `True` if the feature is registered, `False` otherwise.
        """
        return feature in self._features

    def is_extractor_registered(self, extractor):
        """Check if an extractor is present in the registry.

        Parameters
        ----------
        extractor : class
            The feature extractor class to check.

        Returns
        -------
        bool
            `True` if the extractor is registered, `False` otherwise.

        Raises
        ------
        TypeError
            If the class is not a subclass of Extractor.
        """
        self.validate_is_extractor(extractor)

        return extractor in self._extractors

    def extractor_of(self, feature):
        """Get the extractor that extracts a specific feature.

        Parameters
        ----------
        feature : str
            The name of the feature to get the extractor of.

        Returns
        -------
        Extractor
            The feature extractor class that extracts the specified feature.

        Raises
        ------
        EntityNotFoundError
            If the feature is not present in the registry.
        """
        if not self.is_feature_registered(feature):
            raise EntityNotFoundError(f"Feature '{feature}' not found.")

        return self._feature_extractors[feature]

    def extractors_from_data(self, data):
        """Filter the extractors that require the specified data.

        Parameters
        ----------
        data : iterable of str
            The required data to filter extractors by.

        Returns
        -------
        set of Extractor
            The extractors that require all of the specified data.

        Raises
        ------
        RegistryValidationError
            If any of the specified data vectors is invalid.
        """
        invalid_data = set(data).difference(DATAS)
        if invalid_data:
            raise RegistryValidationError(invalid_data)

        return {
            extractor
            for extractor in self._extractors
            if extractor.get_data().issubset(data)
        }

    def extractors_from_features(self, features):
        """Filter the extractors that extract the specified features.

        Parameters
        ----------
        features : iterable of str
            The features to filter extractors by.

        Returns
        -------
        set of Extractor
            The extractors that extract the specified features.

        Raises
        ------
        EntityNotFoundError
            If any of the specified features are not present in the registry.
        """
        extractors = set()
        for feature in features:
            if not self.is_feature_registered(feature):
                raise EntityNotFoundError(f"Feature '{feature}' not found.")
            extractors.add(self._feature_extractors[feature])
        return extractors

    def sort_extractors_by_dependencies(self, extractors):
        """Calculate the feature extractor dependecy resolution order.

        Parameters
        ----------
        extractors : iterable of Extractor
            The extractors to sort by dependencies.

        Returns
        -------
        tuple of Extractor
            The sorted extractors based on their dependencies.

        Raises
        ------
        EntityNotFoundError
            If any of the specified extractors are not present in the registry.
        """
        selected_extractors = []
        features_from_selected = set()
        pending = list(extractors)
        processed = set()

        while pending:
            extractor = pending.pop(0)
            if not self.is_extractor_registered(extractor):
                raise EntityNotFoundError(
                    f"Extractor '{extractor}' not found."
                )
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
        """Generate an execution plan for feature extractors.

        Parameters
        ----------
        data : iterable of str, optional
            The required data for the extractors.
        only : iterable of str, optional
            The features to include in the execution plan.
        exclude : iterable of str, optional
            The features to exclude from the execution plan.

        Returns
        -------
        tuple of Extractor
            The sorted extractors based on the provided constraints.

        Raises
        ------
        RegistryValidationError
            If the same feature is passed in both `only` and `exclude` or if
            any of the specified data vectors in `data` are not valid.
        EntityNotFoundError
            If any of the features passed in `only` or `exclude` are not
            present in the registry.
        """
        if not set(only or []).isdisjoint(exclude or []):
            raise RegistryValidationError(
                "The 'only' and 'exclude' parameters must not overlap."
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
        """frozenset: The extractors registered in the registry."""
        return frozenset(self._extractors)

    @property
    def available_features(self):
        """frozenset: The features available in the registry."""
        return frozenset(self._features)

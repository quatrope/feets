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

"""Features extractors base classes classes"""


# =============================================================================
# IMPORTS
# =============================================================================

import abc
import inspect
import warnings
from dataclasses import dataclass


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_VALUES_TO_REPR = 10

DATA_MAGNITUDE = "magnitude"
DATA_TIME = "time"
DATA_ERROR = "error"
DATA_MAGNITUDE2 = "magnitude2"
DATA_ALIGNED_MAGNITUDE = "aligned_magnitude"
DATA_ALIGNED_MAGNITUDE2 = "aligned_magnitude2"
DATA_ALIGNED_TIME = "aligned_time"
DATA_ALIGNED_ERROR = "aligned_error"
DATA_ALIGNED_ERROR2 = "aligned_error2"

DATAS = (
    DATA_TIME,
    DATA_MAGNITUDE,
    DATA_ERROR,
    DATA_MAGNITUDE2,
    DATA_ALIGNED_TIME,
    DATA_ALIGNED_MAGNITUDE,
    DATA_ALIGNED_MAGNITUDE2,
    DATA_ALIGNED_ERROR,
    DATA_ALIGNED_ERROR2,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ExtractorBadDefinedError(TypeError):
    """The extractor class is not defined properly."""


class ExtractorContractError(ValueError):
    """The extractor doesn't have the expected features, data, parameters
    or whatever.
    """


class ExtractorWarning(UserWarning):
    """Warn about the Extractor behavior."""


class FeatureExtractionWarning(UserWarning):
    """Warn about the calculation of some feature"""


warnings.simplefilter("always", ExtractorWarning)
warnings.simplefilter("always", FeatureExtractionWarning)


# =============================================================================
# EXTRACTOR CONF & UTILS FOR META PROGRAMMING
# =============================================================================


def _isabstract(attr):
    return getattr(attr, "__isabstractmethod__", False)


def _iter_method_parameters(method):
    signature = inspect.signature(method)
    parameters = tuple(signature.parameters.values())[1:]
    return iter(parameters)


@dataclass(frozen=True)
class _ExtractorConf:
    features: frozenset
    optional: frozenset
    required: frozenset
    dependencies: frozenset
    parameters: dict

    @staticmethod
    def _validate_and_add_feature(feature, features, features_attr):
        if not isinstance(feature, str):
            raise ExtractorBadDefinedError(
                f"Feature name must be an instance of string. "
                f"Found {type(feature)}, please check {features_attr!r}"
            )
        if feature in DATAS:
            raise ExtractorBadDefinedError(
                f"Feature can't be in {DATAS!r}. Check {features_attr!r}"
            )
        if feature in features:
            raise ExtractorBadDefinedError(
                f"Duplicated feature {feature!r} in {features_attr!r}"
            )
        features.add(feature)

    @classmethod
    def _get_feature_conf(cls, ecls):
        features_attr = f"{ecls.__qualname__}.features"
        features = set()

        for feature in getattr(ecls, "features", []):
            cls._validate_and_add_feature(feature, features, features_attr)

        if not features:
            raise ExtractorBadDefinedError(
                f"{features_attr!r} must be a non-empty sequence"
            )

        return frozenset(features)

    @staticmethod
    def _validate_and_add_extract_param(
        param, required, optional, dependencies, ecls_name
    ):
        pname = param.name
        has_default = param.default is not param.empty

        if pname in DATAS:
            if has_default:
                optional.add(pname)
            else:
                required.add(pname)
            return

        if has_default:
            raise ExtractorBadDefinedError(
                "Dependencies can't have default values. "
                f"Check {pname!r} in '{ecls_name}.extract()' method"
            )
        dependencies.add(pname)

    @classmethod
    def _get_extract_method_parameters(cls, ecls):
        ecls_name = ecls.__qualname__
        required, optional, dependencies = set(), set(), set()

        extract_params = _iter_method_parameters(ecls.extract)
        for param in extract_params:
            cls._validate_and_add_extract_param(
                param, required, optional, dependencies, ecls_name
            )

        return (
            frozenset(required),
            frozenset(optional),
            frozenset(dependencies),
        )

    @staticmethod
    def _validate_and_add_init_param(param, parameters, ecls_name):
        pname = param.name
        if param.default is param.empty:
            raise ExtractorBadDefinedError(
                f"All parameters in the '{ecls_name}.__init__()' method"
                f"must have a default value. Check {pname!r}."
            )
        parameters[pname] = param.default

    @classmethod
    def _get_init_method_parameters(cls, ecls):
        ecls_name = ecls.__name__
        parameters = {}

        init_params = _iter_method_parameters(ecls.__init__)
        for param in init_params:
            cls._validate_and_add_init_param(param, parameters, ecls_name)

        return dict(parameters)

    @classmethod
    def from_extractor_class(cls, ecls):
        features = cls._get_feature_conf(ecls)
        (
            required,
            optional,
            dependencies,
        ) = cls._get_extract_method_parameters(ecls)
        parameters = cls._get_init_method_parameters(ecls)

        return _ExtractorConf(
            features=features,
            required=required,
            optional=optional,
            dependencies=dependencies,
            parameters=parameters,
        )

    @property
    def data(self):
        return frozenset(self.required.union(self.optional))


# =============================================================================
# EXTRACTOR
# =============================================================================


class Extractor(abc.ABC):

    def __init_subclass__(cls):
        cls_name = cls.__qualname__
        if _isabstract(cls.__init__):
            raise ExtractorBadDefinedError(
                f"'{cls_name}.__init__()' method must be redefined"
            )
        if _isabstract(cls.extract):
            raise ExtractorBadDefinedError(
                f"'{cls_name}.extract()' method must be redefined"
            )

        cls._conf = _ExtractorConf.from_extractor_class(cls)
        del cls.features

    @classmethod
    def get_features(cls):
        """The set of features generated by this extractor."""
        return cls._conf.features

    @classmethod
    def get_data(cls):
        """Retrieve the set of data used for this extractor."""
        return cls._conf.data

    @classmethod
    def get_optional(cls):
        """Retrieve the set of optional data used for this extractor."""
        return cls._conf.optional

    @classmethod
    def get_required_data(cls):
        """Retrieve the required set data used for this extractor."""
        return cls._conf.required

    @classmethod
    def get_dependencies(cls):
        """Which other features are needed to execute this extractor."""
        return cls._conf.dependencies

    @classmethod
    def get_default_params(cls):
        """The default values of the available configuration parameters."""
        return cls._conf.parameters

    def feature_warning(self, msg):
        """Issue a warning."""
        warnings.warn(msg, FeatureExtractionWarning)

    # TO REDEFINE =============================================================

    def __init__(self):
        pass

    @abc.abstractmethod
    def extract(self):
        """Extract features from the time series.

        Returns
        -------
        dict
            The dictionary of features.
        """
        raise NotImplementedError()

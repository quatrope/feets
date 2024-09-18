#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    """The extractor class are not properly defined."""


class ExtractorContractError(ValueError):
    """The extractor don't get the expected features, data, parameters
    or whatever.

    """


class ExtractorWarning(UserWarning):
    """Warn about the Extractor behavior."""


class FeatureExtractionWarning(UserWarning):
    """Warn about calculation of some feature"""


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
    yield from parameters


@dataclass(frozen=True)
class _ExtractorConf:
    features: frozenset
    data: frozenset
    optional: frozenset
    required: frozenset
    dependencies: frozenset
    parameters: dict

    @classmethod
    def _get_features_conf(cls, ecls):
        features_attr = f"{ecls.__qualname__}.features"

        features = set()
        for f in getattr(ecls, "features", []):
            if not isinstance(f, str):
                msg = (
                    "Feature name must be an instance of string. "
                    f"Found {type(f)}, please check {features_attr!r}"
                )
                raise ExtractorBadDefinedError(msg)
            if f in DATAS:
                msg = f"Feature can't be in {DATAS!r}. Check {features_attr!r}"
                raise ExtractorBadDefinedError(msg)
            if f in features:
                msg = f"Duplicated feature {f!r} in {features_attr!r}"
                raise ExtractorBadDefinedError(msg)
            features.add(f)

        if not features:
            msg = f"{features_attr!r} must be a not empty sequence"
            raise ExtractorBadDefinedError(msg)

        return frozenset(features)

    @classmethod
    def _get_extract_method_parameters(cls, ecls):
        cls_name = ecls.__qualname__
        data, required, optional, dependencies = set(), set(), set(), set()

        for param in _iter_method_parameters(ecls.extract):
            pname = param.name
            has_default = not (param.default is param.empty)
            if pname in DATAS:
                data.add(pname)
                if has_default:
                    optional.add(pname)
                else:
                    required.add(pname)
            else:
                if has_default:
                    msg = (
                        "Dependencies with default parameters make no sense. "
                        f"Check {pname!r} in '{cls_name}.extract()' method"
                    )
                    raise ExtractorBadDefinedError(msg)
                dependencies.add(pname)

        return (
            frozenset(data),
            frozenset(required),
            frozenset(optional),
            frozenset(dependencies),
        )

    @classmethod
    def _get_init_method_parameters(cls, ecls):
        cls_name = ecls.__name__
        params = {}
        for param in _iter_method_parameters(ecls.__init__):
            pname = param.name
            has_default = not (param.default is param.empty)
            if not has_default:
                msg = (
                    f"All parameters in the '{cls_name}.__init__()' method "
                    f"must have a default value. Check {pname!r}."
                )
                raise ExtractorBadDefinedError(msg)
            params[pname] = param.default
        return dict(params)

    @classmethod
    def from_extractor_class(cls, ecls):
        features = cls._get_features_conf(ecls)
        (
            data,
            required,
            optional,
            dependencies,
        ) = cls._get_extract_method_parameters(ecls)
        parameters = cls._get_init_method_parameters(ecls)

        conf = _ExtractorConf(
            features=features,
            data=data,
            required=required,
            optional=optional,
            dependencies=dependencies,
            parameters=parameters,
        )

        return conf


# =============================================================================
# EXTRACTOR
# =============================================================================


class Extractor(abc.ABC):

    def __init_subclass__(cls):
        cls_name = cls.__qualname__
        if _isabstract(cls.__init__):
            msg = f"'{cls_name}.__init__()' method must be redefined"
            raise ExtractorBadDefinedError(msg)
        if _isabstract(cls.extract):
            msg = f"'{cls_name}.extract()' method must be redefined"
            raise ExtractorBadDefinedError(msg)

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

    def preprocess_arguments(self, data, dependencies):
        """Preprocess all the incoming arguments \
        (timeserie + dependencies + parameters) to feed the `extract` method.

        """
        kwargs = {}

        # add the required features
        for d in self.get_dependencies():
            kwargs[d] = dependencies[d]

        # add the required data
        for d in self.get_data():
            kwargs[d] = data[d]

        return kwargs

    def postprocess_result(self, result, selected_features):
        """Validate if the extractor generated the expeccted features \
        after calling the `extract` method.

        """

        # validate if the extractor generates the expected features
        expected_features = self.get_features()  # the expected features

        diff = set(result or []).symmetric_difference(
            expected_features or []
        )  # some diff
        if diff:
            cls_name = type(self).__qualname__
            estr, fstr = ", ".join(expected_features), ", ".join(result.keys())
            raise ExtractorContractError(
                f"The extractor '{cls_name}' expected the features {estr}. "
                f"Found: {fstr!r}"
            )

        # todo: normalize `result` to a 1-level dictionary before filtering
        # filter only the selected features
        # selection = set(result or []).intersection(selected_features or [])
        # features = {k: result[k] for k in selection}

        return result

    def select_extract_and_validate(
        self, data, dependencies, selected_features
    ):
        """Internal method designed to select the parameters necessary for
        executing the 'extract()' method, followed by its execution.

        Additionally, finally, check that the features defined in the extractor
        are correctly returned by the 'extract()' method.

        """
        extract_kwargs = self.preprocess_arguments(data, dependencies)

        # run the extractor
        results = self.extract(**extract_kwargs)

        features = self.postprocess_result(results, selected_features)

        return features

    # TO REDEFINE =============================================================

    def __init__(self):
        pass

    @abc.abstractmethod
    def extract(self):
        raise NotImplementedError()


# class MyExtractor(Extractor):
#     features = ["a"]

#     def __init__(self):
#         pass

#     def extract(self, time, magnitude=None):
#         pass


# ext = MyExtractor()

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

import numpy as np

from ..extractor import (
    _ExtractorConf,
    _is_abstract_method,
    Extractor,
    ExtractorBadDefinedError,
)

# =============================================================================
# LIGHT CURVE EXTRACTOR CLASS
# =============================================================================


class LightCurveExtractor(Extractor):
    __abstractclass__ = True

    def __init_subclass__(cls):
        cls.__abstractclass__ = False
        cls_name = cls.__qualname__

        if cls.is_abstract():
            return

        if _is_abstract_method(cls.extract):
            raise ExtractorBadDefinedError(
                f"'{cls_name}.extract()' method must be redefined"
            )

        cls._conf = _ExtractorConf.from_extractor_class(cls)
        del cls.features

    def prepare_extract(self, data, dependencies):
        kwargs = super().prepare_extract(data, dependencies)
        time, magnitude, error = (
            kwargs["time"],
            kwargs["magnitude"],
            kwargs["error"],
        )

        shape = len(time) if time is not None else len(magnitude)

        time = (
            np.arange(shape, dtype=np.float64)
            if time is None
            else np.array(time, dtype=np.float64)
        )
        magnitude = (
            np.zeros(shape, dtype=np.float64)
            if magnitude is None
            else np.array(magnitude, dtype=np.float64)
        )
        error = (
            np.ones(shape, dtype=np.float64)
            if error is None
            else np.array(1 / error**2, dtype=np.float64)
        )
        kwargs["time"], kwargs["magnitude"], kwargs["error"] = (
            time,
            magnitude,
            error,
        )
        return kwargs

    # @abc.abstractmethod
    # def extract(self):
    #     """Extract features from the time series.

    #     Returns
    #     -------
    #     dict
    #         The dictionary of features extracted from the time series.
    #     """
    #     raise NotImplementedError()

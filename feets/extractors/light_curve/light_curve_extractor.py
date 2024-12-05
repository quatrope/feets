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
    DATA_TIME,
    DATA_MAGNITUDE,
    DATA_FLUX,
    DATA_ERROR,
    DATA_FLUX_ERROR,
)
from ...libs import doctools

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

    # API =====================================================================

    @doctools.doc_inherit(Extractor.prepare_extract)
    def prepare_extract(self, data, dependencies):
        time, magnitude, flux, error, flux_error = (
            data.get(DATA_TIME),
            data.get(DATA_MAGNITUDE),
            data.get(DATA_FLUX),
            data.get(DATA_ERROR),
            data.get(DATA_FLUX_ERROR),
        )

        shape = (
            len(time)
            if time is not None
            else len(magnitude) if magnitude is not None else len(flux)
        )

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
        flux = (
            np.zeros(shape, dtype=np.float64)
            if flux is None
            else np.array(flux, dtype=np.float64)
        )
        error = (
            np.ones(shape, dtype=np.float64)
            if error is None
            else np.array(1 / error**2, dtype=np.float64)
        )
        flux_error = (
            np.ones(shape, dtype=np.float64)
            if flux_error is None
            else np.array(1 / flux_error**2, dtype=np.float64)
        )

        data.update(
            {
                DATA_TIME: time,
                DATA_MAGNITUDE: magnitude,
                DATA_FLUX: flux,
                DATA_ERROR: error,
                DATA_FLUX_ERROR: flux_error,
            }
        )

        kwargs = super().prepare_extract(data, dependencies)
        return kwargs

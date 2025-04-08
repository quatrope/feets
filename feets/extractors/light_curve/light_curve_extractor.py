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

        shape = (
            len(data.get(DATA_TIME))
            or data.get(DATA_MAGNITUDE)
            or data.get(DATA_FLUX)
        )
        dtype = np.float64

        preprocessed_data = {
            DATA_TIME: (
                np.arange(shape, dtype=dtype)
                if data.get(DATA_TIME) is None
                else np.array(data.get(DATA_TIME), dtype=dtype)
            ),
            DATA_MAGNITUDE: (
                np.zeros(shape, dtype=dtype)
                if data.get(DATA_MAGNITUDE) is None
                else np.array(data.get(DATA_MAGNITUDE), dtype=dtype)
            ),
            DATA_FLUX: (
                np.zeros(shape, dtype=dtype)
                if data.get(DATA_FLUX) is None
                else np.array(data.get(DATA_FLUX), dtype=dtype)
            ),
            DATA_ERROR: (
                np.ones(shape, dtype=dtype)
                if data.get(DATA_ERROR) is None
                else np.array(1 / data.get(DATA_ERROR)**2, dtype=dtype)
            ),
            DATA_FLUX_ERROR: (
                np.ones(shape, dtype=dtype)
                if data.get(DATA_FLUX_ERROR) is None
                else np.array(1 / data.get(DATA_FLUX_ERROR)**2, dtype=dtype)
            ),
        }

        kwargs = super().prepare_extract(preprocessed_data, dependencies)
        return kwargs

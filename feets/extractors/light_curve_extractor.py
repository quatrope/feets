#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# DOC
# =============================================================================

"""Abstract class for `light_curve` compatible extractors."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import (
    DATA_ERROR,
    DATA_FLUX,
    DATA_FLUX_ERROR,
    DATA_MAGNITUDE,
    DATA_TIME,
    Extractor,
)
from ..libs import doctools

__all__ = ["LightCurveExtractor"]

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DTYPE = np.float64

DATAS_TIME = {DATA_TIME}
DATAS_BRIGHTNESS = {DATA_MAGNITUDE, DATA_FLUX}
DATAS_ERROR = {DATA_ERROR, DATA_FLUX_ERROR}
DATAS = DATAS_TIME.union(DATAS_BRIGHTNESS, DATAS_ERROR)

# =============================================================================
# LIGHT CURVE EXTRACTOR CLASS
# =============================================================================


class LightCurveExtractor(Extractor):
    """Abstract class for `light_curve` compatible extractors."""

    @doctools.doc_inherit(Extractor.__init_subclass__)
    def __init_subclass__(cls):
        super().__init_subclass__()

    # API =====================================================================

    @classmethod
    @doctools.doc_inherit(Extractor.prepare_extract)
    def prepare_extract(cls, data, dependencies):
        # validate and select relevant data and dependencies
        kwargs = super().prepare_extract(data, dependencies)

        shape = next((len(kwargs[k]) for k in kwargs if k in DATAS), 0)
        processed_kwargs = {}

        # select dependencies
        for d in cls.get_dependencies():
            processed_kwargs[d] = kwargs[d]

        # select and format data
        for d in cls.get_data():
            if d in DATAS_TIME:
                processed_kwargs[d] = (
                    np.arange(shape, dtype=DEFAULT_DTYPE)
                    if kwargs.get(d) is None
                    else np.array(kwargs.get(d), dtype=DEFAULT_DTYPE)
                )
            if d in DATAS_BRIGHTNESS:
                processed_kwargs[d] = (
                    np.zeros(shape, dtype=DEFAULT_DTYPE)
                    if kwargs.get(d) is None
                    else np.array(kwargs.get(d), dtype=DEFAULT_DTYPE)
                )
            if d in DATAS_ERROR:
                processed_kwargs[d] = (
                    np.ones(shape, dtype=DEFAULT_DTYPE)
                    if kwargs.get(d) is None
                    else 1 / np.array(kwargs.get(d), dtype=DEFAULT_DTYPE) ** 2
                )

        return processed_kwargs

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Utilities to load, fetch or generate datasets for some example light curves."""

from .base import LightCurve, LightCurveDataset
from .macho import available_MACHO_lc, load_MACHO, load_MACHO_example
from .ogle3 import fetch_OGLE3, load_OGLE3_catalog

__all__ = [
    "available_MACHO_lc",
    "LightCurve",
    "LightCurveDataset",
    "fetch_OGLE3",
    "load_MACHO_example",
    "load_MACHO",
    "load_OGLE3_catalog",
]

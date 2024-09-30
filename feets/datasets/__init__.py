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

"""The :mod:`feets.datasets` module includes utilities to load datasets,
including methods to load and fetch some example light curves.

"""


from .macho import available_MACHO_lc, load_MACHO, load_MACHO_example


__all__ = ["available_MACHO_lc", "load_MACHO", "load_MACHO_example"]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

__doc__ = """All feets tests"""


# =============================================================================
# IMPORTS
# =============================================================================


def run(*args, **kwargs):
    from .run import run  # noqa

    return run(*args, **kwargs)

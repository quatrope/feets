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

import os
import sys

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))


# =============================================================================
# IMPORTS
# =============================================================================


def run(argv=None):
    argv = argv or []
    return pytest.main(["-x", PATH] + argv)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))

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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """Compilation of some of the existing light-curve features."""


# =============================================================================
# CONSTANTS
# =============================================================================

__version__ = ("0", "4")

NAME = "feets"

DOC = __doc__

VERSION = ".".join(__version__)

AUTHORS = "JuanBC"

EMAIL = "jbc.develop@gmail.com"

URL = "http://scikit-criteria.org/"

LICENSE = "MIT"

KEYWORDS = "light curve feature analysis".split()


# =============================================================================
# IMPORTS
# =============================================================================

import os  # noqa

if os.getenv("FEETS_IN_SETUP") != "True":
    from .core import FeatureSpace, MPFeatureSpace  # noqa
    from .extractors import (  # noqa
        Extractor, register_extractor,  # noqa
        registered_extractors, is_registered)  # noqa

del os

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

"""Features extractors classes and register utilities"""

# =============================================================================
# IMPORTS
# =============================================================================

from . import register
from .extractor import (
    DATAS,
    Extractor,
    ExtractorBadDefinedError,
    ExtractorContractError,
    ExtractorWarning,
)
from . import actor


__all__ = [
    "DATAS",
    "ExtractorBadDefinedError",
    "ExtractorContractError",
    "ExtractorWarning",
    "Extractor",
    "register",
    "actor",
]


# =============================================================================
# REGISTERS
# =============================================================================

from .ext_amplitude import *  # noqa
from .ext_anderson_darling import *  # noqa
from .ext_autocor_length import *  # noqa
from .ext_beyond1_std import *  # noqa
from .ext_car import *  # noqa
from .ext_color import *  # noqa
from .ext_con import *  # noqa
from .ext_eta_color import *  # noqa
from .ext_eta_e import *  # noqa
from .ext_flux_percentile_ratio import *  # noqa
from .ext_fourier_components import *  # noqa
from .ext_gskew import *  # noqa
from .ext_linear_trend import *  # noqa
from .ext_lomb_scargle import *  # noqa
from .ext_max_slope import *  # noqa
from .ext_mean import *  # noqa
from .ext_mean_variance import *  # noqa
from .ext_median_abs_dev import *  # noqa
from .ext_median_brp import *  # noqa
from .ext_pair_slope_trend import *  # noqa
from .ext_percent_amplitude import *  # noqa
from .ext_percent_difference_flux_percentile import *  # noqa
from .ext_q31 import *  # noqa
from .ext_rcs import *  # noqa
from .ext_skew import *  # noqa
from .ext_slotted_a_length import *  # noqa
from .ext_small_kurtosis import *  # noqa
from .ext_std import *  # noqa
from .ext_stetson import *  # noqa
from .ext_structure_functions import *  # noqa
from .ext_signature import *  # noqa
from .ext_dmdt import *  # noqa

for cls in register.sort_by_dependencies(Extractor.__subclasses__()):
    register.register_extractor(cls)

del cls

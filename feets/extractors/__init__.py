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

"""Features extractors classes and register utilities"""

# =============================================================================
# IMPORTS
# =============================================================================

from . import delayed
from . import registry
from .extractor import (
    DATAS,
    Extractor,
    ExtractorBadDefinedError,
    ExtractorContractError,
    ExtractorWarning,
)


__all__ = [
    "DATAS",
    "ExtractorBadDefinedError",
    "ExtractorContractError",
    "ExtractorWarning",
    "Extractor",
    "registry",
    "delayed",
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

extractor_registry = registry.ExtractorRegistry()
for cls in Extractor.__subclasses__():
    extractor_registry.register_extractor(cls)

del cls

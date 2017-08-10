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

from __future__ import unicode_literals, print_function


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """Features extractors classes and register utilities"""

__all__ = [
    b"DATAS",
    b"register_extractor",
    b"registered_extractors",
    b"is_registered",
    b"available_features",
    b"extractor_of",
    b"sort_by_dependencies",
    b"ExtractorBadDefinedError",
    b"Extractor"]

# =============================================================================
# IMPORTS
# =============================================================================

import inspect

import six

from .. import err

from .core import Extractor, ExtractorBadDefinedError, DATAS  # noqa


# =============================================================================
# REGISTER UTILITY
# =============================================================================

_extractors = {}


def register_extractor(cls):
    if not inspect.isclass(cls) or not issubclass(cls, Extractor):
        msg = "'cls' must be a subclass of Extractor. Found: {}"
        raise TypeError(msg.format(cls))
    for d in cls.get_dependencies():
        if d not in _extractors.keys():
            msg = "Dependency '{}' from extractor {}".format(d, cls)
            raise ExtractorBadDefinedError(msg)

    _extractors.update((f, cls) for f in cls.get_features())
    return cls


def registered_extractors():
    return dict(_extractors)


def is_registered(obj):
    if isinstance(obj, six.string_types):
        features = [obj]
    elif not inspect.isclass(obj) or not issubclass(obj, Extractor):
        msg = "'cls' must be a subclass of Extractor. Found: {}"
        raise TypeError(msg.format(obj))
    else:
        features = obj.get_features()
    return {f: (f in _extractors) for f in features}


def available_features():
    return _extractors.keys()


def extractor_of(feature):
    return _extractors[feature]


def sort_by_dependencies(exts, retry=100):
    """Calculate the Feature Extractor Resolution Order.

    """
    sorted_ext, features_from_sorted = [], set()
    pending = [(e, 0) for e in exts]
    while pending:
        ext, cnt = pending.pop(0)

        if not isinstance(ext, Extractor) and not issubclass(ext, Extractor):
            msg = "Only Extractor instances are allowed. Found {}."
            raise TypeError(msg.format(type(ext)))

        deps = ext.get_dependencies()
        if deps.difference(features_from_sorted):
            if cnt + 1 > retry:
                msg = "Maximun retry to sort achieved from extractor {}."
                raise RuntimeError(msg.format(type(ext)))
            pending.append((ext, cnt + 1))
        else:
            sorted_ext.append(ext)
            features_from_sorted.update(ext.get_features())
    return tuple(sorted_ext)


# =============================================================================
# REGISTERS
# =============================================================================

from .ext_amplitude import *  # noqa
from .ext_anderson_darling import *  # noqa
from .ext_autocor_length import *  # noqa
from .ext_beyond1_std import *  # noqa
#~ from .ext_car import *  # noqa
#~ from .ext_color import *  # noqa
from .ext_con import *  # noqa
from .ext_eta_color import *  # noqa
from .ext_eta_e import *  # noqa
#~ from .ext_flux_percentile_ratio import *  # noqa
#~ from .ext_fourier_components import *  # noqa
from .ext_gskew import *  # noqa
from .ext_linear_trend import *  # noqa
#~ from .ext_lomb_scargle import *  # noqa
from .ext_max_slope import *  # noqa
from .ext_mean import *  # noqa
from .ext_mean_variance import *  # noqa
from .ext_median_abs_dev import *  # noqa
from .ext_median_brp import *  # noqa
from .ext_pair_slope_trend import *  # noqa
from .ext_percent_amplitude import *  # noqa
#~ from .ext_percent_difference_flux_percentile import *  # noqa
from .ext_q31 import *  # noqa
from .ext_q31 import *  # noqa
from .ext_rcs import *  # noqa
#~ from .ext_signature import *  # noqa
from .ext_skew import *  # noqa
from .ext_slotted_a_length import *  # noqa
from .ext_small_kurtosis import *  # noqa
from .ext_std import *  # noqa
from .ext_stetson import *  # noqa
from .ext_structure_functions import *  # noqa

for cls in sort_by_dependencies(Extractor.__subclasses__()):
    register_extractor(cls)
del cls

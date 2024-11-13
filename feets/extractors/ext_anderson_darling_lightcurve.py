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

"""Anderson-Darling extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import light_curve as lc

import numpy as np

from .extractor import Extractor
from ..libs import doctools

# =============================================================================
# FUNCTIONS
# =============================================================================


def _anderson_darling(magnitude, lightcurve_ext_kwds=None):
    return values[0]


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class AndersonDarlingLightcurve(Extractor):
    """Anderson-Darling extractor.

    **AndersonDarling**

    The Anderson-Darling test is a statistical test of whether a given
    sample of data is drawn from a given probability distribution. When
    applied to testing if a normal distribution adequately describes a set of
    data, it is one of the most powerful statistical tools for detecting most
    departures from normality.

    For a normal distribution the Anderson-Darling statistic should take values
    close to :math:`0.25`.

    References
    ----------
    .. [kim2009trending] Kim, D. W., Protopapas, P., Alcock, C.,
       Byun, Y. I., & Bianco, F. (2009). De-Trending Time Series for
       Astronomical Variability Surveys. Monthly Notices of the Royal
       Astronomical Society, 397(1), 558-568.
       Doi:10.1111/j.1365-2966.2009.14967.x.
    """

    features = {"AndersonDarlingLightcurve"}

    def __init__(self, lightcurve_ext_kwds=None):
        self.lightcurve_ext_kwds = lightcurve_ext_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        time = np.zeros(len(magnitude), dtype=np.float64)
        magnitude = np.array(magnitude, dtype=np.float64)
        feature = lc.AndersonDarlingNormal(**self.lightcurve_ext_kwds)
        [result] = feature(time, magnitude)

        return {"AndersonDarlingLightcurve": result}

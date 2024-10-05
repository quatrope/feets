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

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from scipy import stats

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class AndersonDarling(Extractor):
    """
    **AndersonDarling**

    The Anderson-Darling test is a statistical test of whether a given
    sample of data is drawn from a given probability distribution. When
    applied to testing if a normal distribution adequately describes a set of
    data, it is one of the most powerful statistical tools for detecting most
    departures from normality.

    For a normal distribution the Anderson-Darling statistic should take values
    close to 0.25.


    References
    ----------

    .. [kim2009trending] Kim, D. W., Protopapas, P., Alcock, C.,
       Byun, Y. I., & Bianco, F. (2009). De-Trending Time Series for
       Astronomical Variability Surveys. Monthly Notices of the Royal
       Astronomical Society, 397(1), 558-568.
       Doi:10.1111/j.1365-2966.2009.14967.x.

    """

    features = {"AndersonDarling"}

    def __init__(self):
        self.extractor_warning(
            "The original FATS documentation says that the result of "
            "AndersonDarling must be ~0.25 for gausian distribution but the "
            "result is ~-0.60"
        )

    def extract(self, magnitude):
        ander = stats.anderson(magnitude)[0]
        return {"AndersonDarling": 1 / (1.0 + np.exp(-10 * (ander - 0.3)))}

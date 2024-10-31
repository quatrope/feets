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

"""Mean extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Mean(Extractor):
    r"""Mean extractor.

    **Mean**

    Mean of the magnitudes. For a normal distribution it should take a value
    close to zero.


    Examples
    --------
    Mean of a normal time series:

    >>> fs = feets.FeatureSpace(only=['Mean'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Mean': np.float64(-0.08137835474939399)}

    References
    ----------
    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.
    """

    features = ["Mean"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        mean = np.mean(magnitude)
        return {"Mean": mean}

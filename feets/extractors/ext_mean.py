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

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Mean(Extractor):
    r"""
    **Mean**

    Mean magnitude. For a normal distribution it should take a value
    close to zero:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Mean'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Mean': 0.0082611563419413246}

    References
    ----------

    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    """

    features = ["Mean"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        B_mean = np.mean(magnitude)
        return {"Mean": B_mean}

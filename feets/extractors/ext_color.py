#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

"""Color extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Color(Extractor):
    """Color extractor.

    **Color**

    The color is defined as the difference between the average magnitude of
    two different bands observations.

    References
    ----------
    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["Color"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Color': np.float64(-0.07991933970739044)}
    """

    features = ["Color"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, magnitude2):
        """
        Parameters
        ----------
        magnitude : array_like
        magnitude2 : array_like
        """
        return {"Color": np.mean(magnitude) - np.mean(magnitude2)}

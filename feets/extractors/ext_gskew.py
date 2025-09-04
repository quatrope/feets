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

"""Gskew extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Gskew(Extractor):
    r"""Gskew extractor.

    **Gskew**

    Median-of-magnitudes based measure of the skew.

    .. math::

        Gskew = m_{q3} + m_{q97} - 2m

    Where:

    - :math:`m_{q3}` is the median of magnitudes lesser or equal than the
      quantile 3.
    - :math:`m_{q97}` is the median of magnitudes greater or equal than the
      quantile 97.
    - :math:`m` is the median of magnitudes.
    """

    features = ["Gskew"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        """
        Parameters
        ----------
        magnitude : array-like
        """
        median_mag = np.median(magnitude)
        F_3_value = np.percentile(magnitude, 3)
        F_97_value = np.percentile(magnitude, 97)

        skew = (
            np.median(magnitude[magnitude <= F_3_value])
            + np.median(magnitude[magnitude >= F_97_value])
            - 2 * median_mag
        )
        return {"Gskew": skew}

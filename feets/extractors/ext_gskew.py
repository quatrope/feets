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


class Gskew(Extractor):
    r"""Median-of-magnitudes based measure of the skew.

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

    def __init__(self):
        pass

    def extract(self, magnitude):
        median_mag = np.median(magnitude)
        F_3_value = np.percentile(magnitude, 3)
        F_97_value = np.percentile(magnitude, 97)

        skew = (
            np.median(magnitude[magnitude <= F_3_value])
            + np.median(magnitude[magnitude >= F_97_value])
            - 2 * median_mag
        )
        return {"Gskew": skew}

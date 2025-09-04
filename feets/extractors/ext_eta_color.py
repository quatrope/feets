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

"""Color Eta_e extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class EtaColor(Extractor):
    r"""Color Eta_e extractor.

    **Eta_color** (:math:`\eta_{color}`)

    Variability index Eta_e (:math:`\eta^e`) calculated from the color
    light-curve.

    References
    ----------
    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["Eta_color"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Eta_color': np.float64(0.0007871260219202687)}
    """

    features = ["Eta_color"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, aligned_magnitude, aligned_time, aligned_magnitude2):
        """
        Parameters
        ----------
        aligned_magnitude : array-like
        aligned_time : array-like
        aligned_magnitude2 : array-like
        """

        N = len(aligned_magnitude)
        B_Rdata = aligned_magnitude - aligned_magnitude2

        w = 1.0 / np.power(aligned_time[1:] - aligned_time[:-1], 2)
        w_mean = np.mean(w)

        N = len(aligned_time)
        sigma2 = np.var(B_Rdata)

        S1 = sum(w * (B_Rdata[1:] - B_Rdata[:-1]) ** 2)
        S2 = sum(w)

        eta_B_R = (
            w_mean
            * np.power(aligned_time[N - 1] - aligned_time[0], 2)
            * S1
            / (sigma2 * S2 * N**2)
        )

        return {"Eta_color": eta_B_R}
